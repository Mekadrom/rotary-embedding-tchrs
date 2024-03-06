use einops::einops;
use ndarray::ArrayD;
use tch::{nn::{self, ModuleT}, Device, IndexOp, Kind, Tensor};
use std::f64::consts::PI;

#[macro_export]
macro_rules! tensor {
    ($($x:tt)*) => {
        {
            Tensor::try_from(ndarray::array![$($x)*]).unwrap()
        }
    };
}

#[macro_export]
macro_rules! print_tensor {
    ($x:expr) => {
        println!("{:?}", TryInto::<ArrayD<f64>>::try_into(&$x.to_kind(Kind::Float)).unwrap());
    };

}

/// used for scaling operations; either the scale can be a scalar value, or it can be a tensor containing the same number of values as the tensor to be scaled
pub enum TensorOrScalar<'a> {
    Tensor(&'a Tensor),
    Scalar(&'a f64),
}

pub fn broadcat(tensors: &Vec<Tensor>, dim: Option<&i64>) -> Tensor {
    let broadcasted_tensors = Tensor::broadcast_tensors(&tensors);
    Tensor::cat(&broadcasted_tensors, *dim.unwrap_or(&-1))
}

pub fn rotate_half(x: &Tensor) -> Tensor {
    let x = einops!(".. (d r:2) -> .. d r", x);
    let mut unbound = x.unbind(-1);

    assert!(unbound.len() == 2, "feature dimension must be divisible by 2");

    let x1 = unbound.remove(0);
    let x2 = unbound.remove(0);
    einops!(".. d r -> .. (d r)", &Tensor::stack(&[-x2, x1], -1))
}

pub fn apply_rotary_emb(freqs: &Tensor, t: &Tensor, _start_index: Option<&i64>, _scale: Option<TensorOrScalar>, _seq_dim: Option<&i64>) -> Tensor {
    let seq_dim_or_neg_offset = _seq_dim.unwrap_or(&-2);

    let start_index = _start_index.unwrap_or(&0);
    let scale = _scale.unwrap_or_else(|| TensorOrScalar::Scalar(&1.0));

    let actual_seq_dim;

    if seq_dim_or_neg_offset < &0 {
        actual_seq_dim = t.size().len() as i64 + seq_dim_or_neg_offset;
    } else {
        actual_seq_dim = *seq_dim_or_neg_offset;
    }

    let freqs_mut;
    let narrowed_freqs;

    let freqs_size = freqs.size();
    assert!(freqs_size.len() > 0, "freqs must have at least one dimension");

    if t.size().len() == 3 {
        let seq_len = t.size()[actual_seq_dim as usize];
        let freq_last_dim_len = freqs_size.last().unwrap();
        // Tensor::i is not a drop-in replacement for python slicing
        // in the original, freq = freqs[-seq_len:] will return identity if the first dimension of freqs
        // is less than the length of the sequence (seq_len). so for freqs with batch_size = 1 and shape (1, 4, 8)
        // for example, freqs[-seq_len:] will return shape [1, 4, 8] still for any value seq_len without error.
        // here, we need to make this explicit when using Tensor::i.
        if freq_last_dim_len - seq_len < *freqs.size().first().unwrap() {
            narrowed_freqs = freqs.i((freq_last_dim_len - seq_len)..).to_device(t.device());
            freqs_mut = &narrowed_freqs;
        } else {
            freqs_mut = freqs;
        }
    } else {
        freqs_mut = freqs;
    }

    let rot_dim = *freqs_size.last().unwrap();
    let end_index = start_index + rot_dim;

    assert!(rot_dim <= *t.size().last().unwrap(), "feature dimension is not of sufficient size to rotate in all the positions");

    let t_left = t.narrow(-1, 0, *start_index);
    let mut t_middle = t.narrow(-1, *start_index, end_index - start_index);
    let t_right = t.slice(-1, end_index, t.size().last().map(|&i| i), 1);

    let middle_rotated = &rotate_half(&t_middle);
    
    match scale {
        TensorOrScalar::Tensor(scale) => {
            t_middle = (&t_middle * &freqs_mut.cos() * scale) + (middle_rotated * &freqs_mut.sin() * scale);
        },
        TensorOrScalar::Scalar(scale) => {
            t_middle = (&t_middle * &freqs_mut.cos() * (*scale)) + (middle_rotated * &freqs_mut.sin() * (*scale));
        }
    }
    Tensor::cat(&[t_left, t_middle, t_right], -1)
}

pub fn apply_learned_rotations(rotations: &Tensor, t: &Tensor, _start_index: Option<&i64>, _freq_ranges: Option<&Tensor>) -> Tensor {
    let start_index = _start_index.unwrap_or(&0);

    let result;
    let rearranged;
    if _freq_ranges.is_some() {
        let freq_ranges = _freq_ranges.unwrap();
        let freq_ranges_reshaped = freq_ranges.view([1, 1, -1]);
        let new_rotations = rotations.unsqueeze(-1) * &freq_ranges_reshaped;

        rearranged = einops!(".. d r -> .. (d r)", new_rotations);
        result = Some(&rearranged);
    } else {
        result = None;
    }

    let repeated = einops!(".. n -> .. (n repeat:2)", result.unwrap_or(rotations));
    apply_rotary_emb(&repeated, t, Some(start_index), None, None)
}

#[derive(Debug)]
pub struct RotaryEmbedding<'a> {
    varstore: &'a nn::Path<'a>,
    freqs_for: &'a str,
    cache_if_possible: bool,
    freqs: Tensor,
    learned_freq: bool,
    default_seq_dim: i64,
    interpolate_factor: f64,
    use_xpos: bool,
    scale_base: Option<f64>, // only set if use_xpos is true
}

impl RotaryEmbedding<'_> {
    /// convenience method for creating a new instance of RotaryEmbedding with default values for everything that can be defaulted
    pub fn new_default<'a>(varstore: &'a nn::Path<'a>, dim: i64) -> RotaryEmbedding<'a> {
        RotaryEmbedding::new(
            varstore,
            dim,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// convenience method for creating a new instance of RotaryEmbedding with default values for everything that can be defaulted
    /// and with the use_xpos flag set to true
    pub fn new_xpos<'a>(varstore: &'a nn::Path<'a>, dim: i64, _xpos_scale_base: Option<&'a f64>) -> RotaryEmbedding<'a> {
        RotaryEmbedding::new(
            varstore,
            dim,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&true),
            _xpos_scale_base,
            None,
            None,
            None,
            None,
        )
    }

    pub fn new<'a>(
        varstore: &'a nn::Path<'a>,
        dim: i64,
        _custom_freqs: Option<&'a Tensor>,
        _freqs_for: Option<&'a str>,
        _theta: Option<&'a f64>,
        _max_freq: Option<&'a f64>,
        _num_freqs: Option<&'a i64>,
        _learned_freq: Option<&'a bool>,
        _use_xpos: Option<&'a bool>,
        _xpos_scale_base: Option<&'a f64>,
        _interpolate_factor: Option<&'a f64>,
        _theta_rescale_factor: Option<&'a f64>,
        _seq_before_head_dim: Option<&'a bool>,
        _cache_if_possible: Option<&'a bool>,
    ) -> RotaryEmbedding<'a> {
        let freqs_for = _freqs_for.unwrap_or("lang");
        let theta_rescale_factor = _theta_rescale_factor.unwrap_or(&1.0);

        // proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        // has some connection to NTK literature
        // https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        let theta = &(*_theta.unwrap_or(&10000.0) as f64) * theta_rescale_factor.powf((dim as f64 / (dim as f64 - 2.)) as f64);

        let max_freq = _max_freq.unwrap_or(&10.0);
        let num_freqs = _num_freqs.unwrap_or(&1);
        let learned_freq = _learned_freq.unwrap_or(&false);
        let use_xpos = _use_xpos.unwrap_or(&false);
        let xpos_scale_base = _xpos_scale_base.unwrap_or(&512.);
        let interpolate_factor = _interpolate_factor.unwrap_or(&1.0);
        let seq_before_head_dim = _seq_before_head_dim.unwrap_or(&false);
        let cache_if_possible = _cache_if_possible.unwrap_or(&true);

        let freqs_tensor;

        if _custom_freqs.is_none() {
            match _freqs_for {
                Some("pixel") => {
                    freqs_tensor = Tensor::linspace(1.0, max_freq / 2.0, dim / 2, (Kind::Double, varstore.device())) * PI;
                },
                Some("constant") => {
                    freqs_tensor = Tensor::ones(vec![*num_freqs], (Kind::Double, varstore.device()));
                },
                _ => {
                    let range = Tensor::arange_start_step(0, dim, 2, (Kind::Double, Device::Cpu));
                    freqs_tensor = 1.0 / Tensor::pow_scalar(theta, &(range / (dim as f64))).to_kind(Kind::Double);
                }
            }
        } else {
            freqs_tensor = _custom_freqs.unwrap().shallow_clone();
        }

        let freqs = varstore.var_copy("freqs", &freqs_tensor);
        let default_seq_dim = if *seq_before_head_dim { -3 } else { -2 };

        assert!(*interpolate_factor >= 1.0, "interpolate factor must be greater than or equal to 1.0");

        let instance = RotaryEmbedding {
            varstore,
            freqs_for,
            cache_if_possible: *cache_if_possible,
            freqs,
            learned_freq: *learned_freq,
            default_seq_dim: default_seq_dim,
            interpolate_factor: *interpolate_factor,
            use_xpos: *use_xpos,
            scale_base: if *use_xpos { Some(*xpos_scale_base) } else { None }
        };

        if !use_xpos {
            RotaryEmbedding::tmp_store(&instance, "scale", None);
            return instance;
        }

        RotaryEmbedding::tmp_store(&instance, "cached_freqs", None);
        RotaryEmbedding::tmp_store(&instance, "cached_scales", None);
        RotaryEmbedding::tmp_store(&instance, "dummy", Some(&Tensor::from(0.)));

        let scale = (Tensor::arange_start_step(0, dim, 2, (Kind::Double, varstore.device())) + (0.4 * dim as f64)) / (1.4 * dim as f64);

        print_tensor!(scale);
        RotaryEmbedding::tmp_store(&instance, "scale", Some(&scale));

        instance
    }

    fn tmp_store(&self, key: &str, value: Option<&Tensor>) -> Option<Tensor> {
        if value.is_none() {
            return None;
        }
        Some(self.varstore.var_copy(key, value.unwrap()))
    }

    fn get_seq_pos(&self, seq_len: &i64, kind: &Kind, device: &Device, _offset: Option<&i64>) -> Tensor {
        let offset = _offset.unwrap_or(&0);
        (Tensor::arange(*seq_len, (*kind, *device)) + *offset as f64) / self.interpolate_factor
    }

    pub fn rotate_queries_or_keys(&self, t: &Tensor, _seq_dim: Option<&i64>, _offset: Option<&i64>, _freq_seq_len: Option<&i64>) -> Tensor {
        let seq_dim_or_neg_offset = _seq_dim.unwrap_or(&self.default_seq_dim);
        let offset = _offset.unwrap_or(&0);

        assert!(!self.use_xpos, "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings");

        let device = t.device();
        let kind = t.kind();

        let actual_seq_dim;

        if seq_dim_or_neg_offset < &0 {
            actual_seq_dim = t.size().len() as i64 + seq_dim_or_neg_offset
        } else {
            actual_seq_dim = *seq_dim_or_neg_offset;
        }

        let mut seq_len = t.size()[actual_seq_dim as usize];

        if _freq_seq_len.is_some() {
            let freq_seq_len = _freq_seq_len.unwrap();
            assert!(freq_seq_len >= &seq_len);
            seq_len = *freq_seq_len;
        }

        let seq_pos = &self.get_seq_pos(
            &seq_len,
            &kind,
            &device,
            Some(offset)
        );

        let mut freqs = self.forward(
            seq_pos,
            Some(&seq_len),
            Some(offset)
        );

        if actual_seq_dim == t.size().len() as i64 - 3{
            freqs = freqs.unsqueeze(1);
        }

        apply_rotary_emb(&freqs, t, None, None, Some(&actual_seq_dim))
    }

    pub fn rotate_queries_with_cached_keys(&self, q: &Tensor, k: &Tensor, _seq_dim: Option<&i64>, _offset: Option<&i64>) -> (Tensor, Tensor) {
        let seq_dim_or_neg_offset = _seq_dim.unwrap_or(&self.default_seq_dim);

        let q_size = q.size().len() as i64;
        let k_size = k.size().len() as i64;

        // accounts for both negative offset dimension indices and regular indices
        let actual_queries_seq_dim;
        let actual_keys_seq_dim;
        if seq_dim_or_neg_offset < &0 {
            actual_queries_seq_dim = q_size + seq_dim_or_neg_offset;
            actual_keys_seq_dim = k_size + seq_dim_or_neg_offset;
        } else {
            actual_queries_seq_dim = *seq_dim_or_neg_offset;
            actual_keys_seq_dim = *seq_dim_or_neg_offset;
        }

        let q_len = q.size()[actual_queries_seq_dim as usize];
        let k_len = k.size()[actual_keys_seq_dim as usize];

        assert!(q_len <= k_len, "query length must be less than or equal to key length");

        let rotated_q = self.rotate_queries_or_keys(q, Some(seq_dim_or_neg_offset), None, Some(&k_len));
        let rotated_k = self.rotate_queries_or_keys(k, Some(seq_dim_or_neg_offset), None, None);

        (rotated_q, rotated_k)
    }

    pub fn rotate_queries_and_keys(&self, q: &Tensor, k: &Tensor, _seq_dim: Option<&i64>) -> (Tensor, Tensor) {
        let seq_dim_or_neg_offset = _seq_dim.unwrap_or(&self.default_seq_dim);

        assert!(self.use_xpos, "you must use `.rotate_queries_or_keys` method instead and pass in only queries, for non-length extrapolatable rotary embeddings");
        let device = q.device();
        let kind = q.kind();

        let actual_seq_dim;
        if seq_dim_or_neg_offset < &0 {
            actual_seq_dim = q.size().len() as i64 + seq_dim_or_neg_offset;
        } else {
            actual_seq_dim = *seq_dim_or_neg_offset;
        }
        let seq_len = q.size()[actual_seq_dim as usize];

        let seq = self.get_seq_pos(&seq_len, &kind, &device, None);

        let mut freqs = self.forward(&seq, Some(&seq_len), None);
        let mut scale = self.get_scale(&seq, Some(&seq_len), None).to_kind(kind);

        if actual_seq_dim == q.size().len() as i64 - 3 {
            freqs = freqs.unsqueeze(1);
            scale = scale.unsqueeze(1);
        }

        let mut rotated_q = apply_rotary_emb(&freqs, q, None, Some(TensorOrScalar::Tensor(&scale)), Some(&actual_seq_dim));
        let mut rotated_k = apply_rotary_emb(&freqs, k, None, Some(TensorOrScalar::Tensor(&scale.pow_(-1))), Some(&actual_seq_dim));

        rotated_q = rotated_q.to_kind(q.kind());
        rotated_k = rotated_k.to_kind(k.kind());

        (rotated_q, rotated_k)
    }

    pub fn get_scale(&self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor {
        assert!(self.use_xpos);

        let seq_len = _seq_len.unwrap_or(&0);
        let offset = _offset.unwrap_or(&0);

        let should_cache = self.cache_if_possible && _seq_len.is_some();

        let _cached_scales = self.varstore.get("cached_scales");

        if should_cache && _cached_scales.is_some() {
            let cached_scales = _cached_scales.unwrap();
            if (seq_len + offset) <= cached_scales.size()[0 as usize] {
                return cached_scales.i(*offset..(offset + seq_len));
            }
        }

        // scale base will always be present here because of the self.use_xpos assertion at the beginning of this function
        let power = (t - t.size()[0] as f64 / 2.) / self.scale_base.unwrap();
        let scale = self.varstore.get("scale").unwrap().pow(&power.unsqueeze(-1));
        let scale_cat = Tensor::cat(&vec![&scale, &scale], -1).to_kind(scale.kind());

        if should_cache {
            self.tmp_store("cached_scales", Some(&scale_cat));
        }

        scale_cat
    }

    pub fn get_axial_freqs(&self, dims: &[i64]) -> Tensor {
        let mut all_freqs = Vec::new();

        for (ind, &dim) in dims.iter().enumerate() {
            let pos = if self.freqs_for == "pixel" {
                Tensor::linspace(-1.0, 1.0, dim, (Kind::Float, self.varstore.device()))
            } else {
                Tensor::arange(dim, (Kind::Int64, self.varstore.device()))
            };

            let freqs = self.forward(&pos, Some(&dim), None).to_kind(Kind::Double);

            // construct new shape for freqs with additional dimensions
            let freq_size = freqs.size();
            let num_freqs_dims = freq_size.len();
            let freqs_last_dim_size = freq_size.last().unwrap();
            
            let mut new_shape = Vec::new();
            let mut cur_freq_dim = 0;

            // for every dimension of freqs before the last (dims.len() + 1) dimensions, include the entire dimension
            if num_freqs_dims > dims.len() + 1 { 
                for _ in 0..(num_freqs_dims - (dims.len() + 1)) {
                    new_shape.push(freq_size[cur_freq_dim]);
                    cur_freq_dim += 1;
                }
            }

            // insert new dimensions where necessary, but where the current index in dims array is, insert the size of the frequency dimension
            for i in 0..dims.len() {
                if i == ind {
                    new_shape.push(freq_size[cur_freq_dim]);
                    cur_freq_dim += 1;
                } else {
                    new_shape.push(1);
                }
            }

            new_shape.push(*freqs_last_dim_size);

            let freqs_reshaped = freqs.view(new_shape.as_slice());
            all_freqs.push(freqs_reshaped);
        }

        let broadcasted_freqs = Tensor::broadcast_tensors(&all_freqs);
        Tensor::cat(&broadcasted_freqs, -1)
    }
}

trait ModuleTII {
    fn forward(&self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor;
}

impl ModuleTII for RotaryEmbedding<'_> {
    fn forward(&self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor {
        let offset = _offset.unwrap_or(&0);

        let should_cache = self.cache_if_possible && !self.learned_freq && _seq_len.is_some() && self.freqs_for != "pixel";

        println!("should_cache: {}", should_cache);

        if should_cache {
            let seq_len = _seq_len.unwrap();
            println!("seq_len: {}", seq_len);
            println!("offset: {}", offset);
            let _cached_freqs = self.varstore.get("cached_freqs");
            println!("_cached_freqs: {}", _cached_freqs.is_some());
            if _cached_freqs.is_some() {
                let cached_freqs = _cached_freqs.unwrap();
                if (offset + seq_len) <= cached_freqs.size()[0] {
                    println!("cached_freqs.size()[0]: {:?}", cached_freqs.size()[0]);
                    return cached_freqs.i(*offset..(offset + seq_len)).detach();
                }
            }
        }

        let freqs_reshaped = self.freqs.view([1, -1]);
        let freqs_einsum = t.to_kind(self.freqs.kind()).unsqueeze(-1) * &freqs_reshaped;
        let freqs_repeated = einops!(".. n -> .. (n repeat:2)", &freqs_einsum);

        if should_cache {
            self.tmp_store("cached_freqs", Some(&freqs_repeated.detach()));
        }

        freqs_repeated
    }
}

impl ModuleT for RotaryEmbedding<'_> {
    fn forward_t(&self, t: &Tensor, _train: bool) -> Tensor {
        self.forward(t, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;

    static DIM: i64 = 8;
    static THETA: f64 = 10000.0;

    lazy_static! {
        // shape is 2, 4, 8
        static ref Q_OR_K: ndarray::Array3<f64> = ndarray::array![
            [[0., 1., 2., 3., 4., 5., 6., 7.], 
             [1., 2., 3., 4., 5., 6., 7., 8.], 
             [2., 3., 4., 5., 6., 7., 8., 9.], 
             [3., 4., 5., 6., 7., 8., 9., 0.]],
            [[0., 1., 2., 3., 4., 5., 6., 7.], 
             [1., 2., 3., 4., 5., 6., 7., 8.], 
             [2., 3., 4., 5., 6., 7., 8., 9.], 
             [3., 4., 5., 6., 7., 8., 9., 0.]],
        ];
    }

    fn q_or_k() -> Tensor {
        Tensor::try_from(Q_OR_K.to_owned().into_dyn()).unwrap().to_kind(Kind::Double)
    }

    fn assert_tensors_eq(actual: &Tensor, expected: &Tensor) {
        assert_eq!(expected.kind(), actual.kind(), "{}", format!("expected {:?}, but got {:?}", expected.kind(), actual.kind()));
        assert_eq!(expected.size(), actual.size(), "{}", format!("expected {:?}, but got {:?}", expected.size(), actual.size()));
        assert!(expected.allclose(actual, 1e-5, 1e-5, false), "{}", format!("expected \n{}, but got \n{}. \n difference: \n{}", expected, actual, expected-actual));
    }

    fn get_freqs() -> Tensor {
        let range = Tensor::arange_start_step(0, DIM, 2, (Kind::Float, Device::Cpu));
        1.0 / Tensor::pow_scalar(THETA, &(range / (DIM as f64)))
    }

    #[test]
    fn test_broadcat1() {
        let actual = broadcat(&vec![tensor![1., 2., 3., 4.], tensor![4., 3., 2., 1.]], None);
        let expected = tensor![1., 2., 3., 4., 4., 3., 2., 1.];
        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_broadcat2() {
        let actual = broadcat(&vec![tensor![[1., 2., 3., 4.], [4., 3., 2., 1.]], tensor![[5., 6., 7., 8.], [8., 7., 6., 5.]]], None);
        let expected = tensor![[1., 2., 3., 4., 5., 6., 7., 8.], [4., 3., 2., 1., 8., 7., 6., 5.]];
        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_broadcat3() {
        let actual = broadcat(&vec![tensor![[1., 2., 3., 4.], [4., 3., 2., 1.]], tensor![[5., 6., 7., 8.], [8., 7., 6., 5.]]], Some(&0));
        let expected = tensor![[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.], [8., 7., 6., 5.]];
        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_rotary_emb1() {
        let q_or_k = q_or_k(); // shape: 2, 4, 8

        let freqs_tensor = get_freqs();

        let actual = apply_rotary_emb(&freqs_tensor, &q_or_k, None, None, None);
        let expected = tensor![[[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
            3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000],
          [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
            4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
            7.0000000000000000,  8.0000000000000000],
          [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
            5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
            8.0000000000000000,  9.0000000000000000],
          [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
            6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
            9.0000000000000000,  0.0000000000000000]],
         [[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
            3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000],
          [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
            4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
            7.0000000000000000,  8.0000000000000000],
          [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
            5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
            8.0000000000000000,  9.0000000000000000],
          [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
            6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
            9.0000000000000000,  0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);

        let q_or_k_squoze = q_or_k.i(0).squeeze_dim(0); // shape: [4, 8]

        let actual = apply_rotary_emb(&freqs_tensor, &q_or_k_squoze, None, None, None);
        let expected_squoze = tensor![[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
            3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000],
          [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
            4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
            7.0000000000000000,  8.0000000000000000],
          [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
            5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
            8.0000000000000000,  9.0000000000000000],
          [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
            6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
            9.0000000000000000,  0.0000000000000000]];

        assert_tensors_eq(&actual, &expected_squoze);

        let q_or_k = q_or_k.unsqueeze(0).repeat(&[2, 1, 1, 1]);

        let actual = apply_rotary_emb(&freqs_tensor, &q_or_k, None, None, None);
        let expected = expected.unsqueeze(0).repeat(&[2, 1, 1, 1]);

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    #[should_panic]
    fn test_apply_rotary_emb_panics() {
        let q_or_k = tensor![
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ]; // shape: [2, 4, 2]

        let freqs_tensor = get_freqs();

        // should panic because feature dimension is not of sufficient size to rotate in all the positions for dim = 8
        // which is obtained from freqs_tensor
        let _actual = apply_rotary_emb(&freqs_tensor, &q_or_k, None, None, None);
    }

    #[test]
    fn test_apply_rotary_emb_scalar_scale() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let freqs_tensor = get_freqs();

        let actual = apply_rotary_emb(&freqs_tensor, &q_or_k, None, Some(TensorOrScalar::Scalar(&(4./3.))), None);
        let expected = tensor![[[-1.1219613552093506,  1.3266723155975342,  2.6265342235565186,
            4.0026645660400391,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000],
        [-1.5235195159912109,  2.7864558696746826,  3.9464678764343262,
            5.3373312950134277,  5.0000000000000000,  6.0000000000000000,
            7.0000000000000000,  8.0000000000000000],
        [-1.9250775575637817,  4.2462391853332520,  5.2664012908935547,
            6.6719970703125000,  6.0000000000000000,  7.0000000000000000,
            8.0000000000000000,  9.0000000000000000],
        [-2.3266358375549316,  5.7060227394104004,  6.5863351821899414,
            8.0066633224487305,  7.0000000000000000,  8.0000000000000000,
            9.0000000000000000,  0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_rotary_emb_tensor_scale() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let scale = tensor![4./3., 3./2., 2./1., 1./2.];

        let actual = apply_rotary_emb(&get_freqs(), &q_or_k, None, Some(TensorOrScalar::Tensor(&scale)), None);
        let expected = tensor![[[-1.12196135520935058594,  1.49250626564025878906,
            3.93980097770690917969,  1.50099933147430419922,
            4.00000000000000000000,  5.00000000000000000000,
            6.00000000000000000000,  7.00000000000000000000],
          [-1.52351951599121093750,  3.13476276397705078125,
            5.91970157623291015625,  2.00149893760681152344,
            5.00000000000000000000,  6.00000000000000000000,
            7.00000000000000000000,  8.00000000000000000000],
          [-1.92507755756378173828,  4.77701950073242187500,
            7.89960145950317382812,  2.50199890136718750000,
            6.00000000000000000000,  7.00000000000000000000,
            8.00000000000000000000,  9.00000000000000000000],
          [-2.32663583755493164062,  6.41927528381347656250,
            9.87950229644775390625,  3.00249862670898437500,
            7.00000000000000000000,  8.00000000000000000000,
            9.00000000000000000000,  0.00000000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_rotary_emb_seq_dim_first() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).permute(&[1, 0, 2]); // shape: [4, 1, 8]

        let actual = apply_rotary_emb(&get_freqs(), &q_or_k, None, None, Some(&0));
        let expected = tensor![[[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
                3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000]],
            [[-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
                4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
                7.0000000000000000,  8.0000000000000000]],
            [[-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
                5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
                8.0000000000000000,  9.0000000000000000]],
            [[-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
                6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
                9.0000000000000000,  0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_rotary_emb_diff_start_index() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let actual = apply_rotary_emb(&get_freqs(), &q_or_k, Some(&4), None, None);
        let expected = tensor![[[ 0.0000000000000000e+00,  1.0000000000000000e+00,
                2.0000000000000000e+00,  3.0000000000000000e+00,
            -2.0461452007293701e+00,  5.3743543624877930e+00,
                5.9297013282775879e+00,  7.0059967041015625e+00],
            [ 1.0000000000000000e+00,  2.0000000000000000e+00,
                3.0000000000000000e+00,  4.0000000000000000e+00,
            -2.3473141193389893e+00,  6.4691920280456543e+00,
                6.9196515083312988e+00,  8.0069961547851562e+00],
            [ 2.0000000000000000e+00,  3.0000000000000000e+00,
                4.0000000000000000e+00,  5.0000000000000000e+00,
            -2.6484827995300293e+00,  7.5640296936035156e+00,
                7.9096012115478516e+00,  9.0079965591430664e+00],
            [ 3.0000000000000000e+00,  4.0000000000000000e+00,
                5.0000000000000000e+00,  6.0000000000000000e+00,
            -2.9496512413024902e+00,  8.6588668823242188e+00,
                8.9995498657226562e+00,  8.9999996125698090e-03]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_learned_rotations() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let rotations = tensor![
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ];

        let actual = apply_learned_rotations(&rotations, &q_or_k, None, None);
        let expected = tensor![[[ 0.0000000000000000,  1.0000000000000000, -0.4631552696228027,
                3.5756800174713135,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000],
            [-1.1426396369934082,  1.9220756292343140, -4.3939332962036133,
                2.3860745429992676,  5.0000000000000000,  6.0000000000000000,
                7.0000000000000000,  8.0000000000000000],
            [-3.5601859092712402,  0.5701543092727661, -5.7051134109497070,
            -2.9071772098541260,  6.0000000000000000,  7.0000000000000000,
                8.0000000000000000,  9.0000000000000000],
            [-3.5344574451446533, -3.5366101264953613,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
                9.0000000000000000,  0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_learned_rotations_freq_ranges() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let rotations = tensor![
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ]; // shape: [1, 4, 2]

        let freq_ranges = tensor![0, 1];

        let actual = apply_learned_rotations(&rotations, &q_or_k, None, Some(&freq_ranges));
        let expected = tensor![[[  0.0000000000000000,   1.0000000000000000,   2.0000000000000000,
                3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
            -0.0754923820495605,   9.2192354202270508],
            [  1.0000000000000000,   2.0000000000000000,  -1.7449767589569092,
                4.6856222152709961,   5.0000000000000000,   6.0000000000000000,
            -8.9333658218383789,   5.7615070343017578],
            [  2.0000000000000000,   3.0000000000000000,  -6.2110743522644043,
                1.5564553737640381,   6.0000000000000000,   7.0000000000000000,
            -10.9981079101562500,  -4.9032244682312012],
            [  3.0000000000000000,   4.0000000000000000,  -5.7966823577880859,
            -5.2343549728393555,   7.0000000000000000,   8.0000000000000000,
                9.0000000000000000,   0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_apply_learned_rotations_start_index() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let rotations = tensor![
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ]; // shape: [1, 4, 2]

        let actual = apply_learned_rotations(&rotations, &q_or_k, Some(&4), None);
        let expected = tensor![[[  0.0000000000000000,   1.0000000000000000,   2.0000000000000000,
                3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
            -0.0754923820495605,   9.2192354202270508],
            [  1.0000000000000000,   2.0000000000000000,   3.0000000000000000,
                4.0000000000000000,  -2.3473141193389893,   7.4491686820983887,
            -8.9333658218383789,   5.7615070343017578],
            [  2.0000000000000000,   3.0000000000000000,   4.0000000000000000,
                5.0000000000000000,  -8.8619632720947266,   2.5427563190460205,
            -10.9981079101562500,  -4.9032244682312012],
            [  3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
                6.0000000000000000,  -8.0589075088500977,  -6.9320998191833496,
                9.0000000000000000,   0.0000000000000000]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_default_rotary_embedding_rotate_queries_or_keys() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);
        
        let actual = rotary_embedding.rotate_queries_or_keys(&q_or_k, None, None, None);
        let expected = tensor![[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
                3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000],
            [-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
                4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
                6.9919967651367188,  8.0069961547851562],
            [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
                5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
                7.9819841384887695,  9.0159816741943359],
            [-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
                7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
                8.9999599456787109,  0.0269999597221613]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_different_seq_dim_rotary_embedding_rotate_queries_or_keys() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).permute(&[1, 0, 2]); // shape: [4, 1, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let actual = rotary_embedding.rotate_queries_or_keys(&q_or_k, Some(&-3), None, None);
        let expected = tensor![[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
                3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000]],
            [[-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
                4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
                6.9919967651367188,  8.0069961547851562]],
            [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
                5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
                7.9819841384887695,  9.0159816741943359]],
            [[-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
                7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
                8.9999599456787109,  0.0269999597221613]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_different_offset_rotary_embedding_rotate_queries_or_keys() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let actual = rotary_embedding.rotate_queries_or_keys(&q_or_k, None, Some(&4), None);
        let expected = tensor![[[ 0.7568024992942810, -0.6536436080932617,  0.6738668680191040,
                3.5420196056365967,  3.7968537807464600,  5.1559576988220215,
                5.9719524383544922,  7.0239443778991699],
            [ 2.2015109062194824, -0.3915998935699463,  0.7150454521179199,
                4.9486069679260254,  4.6938767433166504,  6.2423977851867676,
                6.9599123001098633,  8.0348997116088867],
            [ 2.7585868835449219,  2.3216798305511475,  0.4781301021575928,
                6.3852481842041016,  5.5694556236267090,  7.3471879959106445,
                7.9458560943603516,  9.0478372573852539],
            [-0.3662395477294922,  4.9865689277648926, -0.0410947799682617,
                7.8101415634155273,  6.4233145713806152,  8.4700078964233398,
                8.9997797012329102,  0.0629994869232178]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_freq_seq_len_ovr_valid_same_len_rotary_embedding_rotate_queries_or_keys() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let actual = rotary_embedding.rotate_queries_or_keys(&q_or_k, None, None, Some(&8));
        let expected = tensor![[[ 0.7568024992942810, -0.6536436080932617,  0.6738668680191040,
                3.5420196056365967,  3.7968537807464600,  5.1559576988220215,
                5.9719524383544922,  7.0239443778991699],
            [ 2.2015109062194824, -0.3915998935699463,  0.7150454521179199,
                4.9486069679260254,  4.6938767433166504,  6.2423977851867676,
                6.9599123001098633,  8.0348997116088867],
            [ 2.7585868835449219,  2.3216798305511475,  0.4781301021575928,
                6.3852481842041016,  5.5694556236267090,  7.3471879959106445,
                7.9458560943603516,  9.0478372573852539],
            [-0.3662395477294922,  4.9865689277648926, -0.0410947799682617,
                7.8101415634155273,  6.4233145713806152,  8.4700078964233398,
                8.9997797012329102,  0.0629994869232178]]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    #[should_panic]
    fn test_freq_seq_len_ovr_invalid_lt_rotary_embedding_rotate_queries_or_keys() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        // sequence length 2, feature dimension 8
        let _actual = rotary_embedding.rotate_queries_or_keys(&q_or_k, None, None, Some(&2));
    }

    #[test]
    #[should_panic]
    fn test_rotary_embedding_rotate_queries_or_keys_throws_xpos() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let xpos_rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        // should panic because rotate_queries_or_keys does not work for length extrapolatable rotary embeddings
        let _actual = xpos_rotary_embedding.rotate_queries_or_keys(&q_or_k, None, None, None);
    }

    #[test]
    fn test_rotary_embedding_rotate_queries_with_cached_keys() {
        let queries = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]
        let keys = tensor![
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]]
        ];

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let (actual_queries, actual_keys) = rotary_embedding.rotate_queries_with_cached_keys(&queries, &keys, None, None);

        let expected_queries = tensor![[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
                3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000],
            [-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
                4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
                6.9919967651367188,  8.0069961547851562],
            [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
                5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
                7.9819841384887695,  9.0159816741943359],
            [-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
                7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
                8.9999599456787109,  0.0269999597221613]]];
        let expected_keys = tensor![[[ 3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
                6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
                9.0000000000000000,  0.0000000000000000],
            [-1.4438081979751587,  3.3038489818572998,  3.4808495044708252,
                5.3743543624877930,  5.9297013282775879,  7.0596489906311035,
                7.9909963607788086,  9.0079965591430664],
            [-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
                4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
                6.9839863777160645,  8.0139846801757812],
            [-0.1411200016736984, -0.9899924993515015,  1.0241123437881470,
                3.4570498466491699,  3.8482227325439453,  5.1177320480346680,
                5.9789733886718750,  7.0179686546325684]]];

        assert_tensors_eq(&actual_queries, &expected_queries);
        assert_tensors_eq(&actual_keys, &expected_keys);
    }

    #[test]
    fn test_different_seq_dim_rotary_embedding_rotate_queries_with_cached_keys() {
        let queries = q_or_k().i(0).unsqueeze(0).permute(&[1, 0, 2]); // shape: [4, 1, 8]
        let keys = tensor![
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]]
        ].permute(&[1, 0, 2]); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        // specifying sequence dimension causes caching
        let (actual_queries, actual_keys) = rotary_embedding.rotate_queries_with_cached_keys(&queries, &keys, Some(&-3), None);

        let expected_queries = tensor![[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
            3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000]],
         [[-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
            4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
            6.9919967651367188,  8.0069961547851562]],
         [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
            5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
            7.9819841384887695,  9.0159816741943359]],
         [[-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
            7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
            8.9999599456787109,  0.0269999597221613]]];
         let expected_keys = tensor![[[ 3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
            6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
            9.0000000000000000,  0.0000000000000000]],
         [[-1.4438081979751587,  3.3038489818572998,  3.4808495044708252,
            5.3743543624877930,  5.9297013282775879,  7.0596489906311035,
            7.9909963607788086,  9.0079965591430664]],
         [[-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
            4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
            6.9839863777160645,  8.0139846801757812]],
         [[-0.1411200016736984, -0.9899924993515015,  1.0241123437881470,
            3.4570498466491699,  3.8482227325439453,  5.1177320480346680,
            5.9789733886718750,  7.0179686546325684]]];

        assert_tensors_eq(&actual_queries, &expected_queries);
        assert_tensors_eq(&actual_keys, &expected_keys);

        // works with seq_dim being an absolute value, unlike the original python implementation
        let (actual_queries, actual_keys) = rotary_embedding.rotate_queries_with_cached_keys(&queries, &keys, Some(&0), None);

        assert_tensors_eq(&actual_queries, &expected_queries);
        assert_tensors_eq(&actual_keys, &expected_keys);
    }

    #[test]
    fn test_rotary_embedding_rotate_queries_and_keys() {
        let queries = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]
        let keys = tensor![
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]]
        ];

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let (actual_queries, actual_keys) = rotary_embedding.rotate_queries_and_keys(&queries, &keys, None);

        let expected_queries = tensor![[[ 0.0000000000000000,  1.0030015707015991,  2.0034546852111816,
                3.0023059844970703,  4.0196223258972168,  5.0150079727172852,
                6.0103640556335449,  7.0053806304931641],
            [-1.1454386711120605,  1.9249579906463623,  2.5879111289978027,
                4.2811617851257324,  4.9518523216247559,  6.0587716102600098,
                6.9980325698852539,  8.0100736618041992],
            [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
                5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
                7.9819841384887695,  9.0159816741943359],
            [-3.5258200168609619, -3.5313138961791992,  3.0009701251983643,
                7.2068500518798828,  6.7403740882873535,  8.1940803527832031,
                8.9921970367431641,  0.0269895885139704]]];
        let expected_keys = tensor![[[ 2.9853551387786865,  3.9880297183990479,  4.9913783073425293,
            5.9953918457031250,  6.9658288955688477,  7.9760594367980957,
            8.9844808578491211,  0.0000000000000000],
          [-1.4402798414230347,  3.2989020347595215,  3.4778468608856201,
            5.3722896575927734,  5.9152107238769531,  7.0490779876708984,
            7.9841032028198242,  9.0045347213745117],
          [-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
            4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
            6.9839863777160645,  8.0139846801757812],
          [-0.1414657086133957, -0.9914771318435669,  1.0249965190887451,
            3.4583785533905029,  3.8576498031616211,  5.1254072189331055,
            5.9841351509094238,  7.0206656455993652]]];

        assert_tensors_eq(&actual_queries, &expected_queries);
        assert_tensors_eq(&actual_keys, &expected_keys);
    }

    #[test]
    #[should_panic]
    fn test_rotary_embedding_rotate_queries_and_keys_throws_no_xpos() {
        let queries = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]
        let keys = tensor![
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]]
        ];

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        // throws because use_xpos needs to be true to call rotate_queries_and_keys
        let (_actual_queries, _actual_keys) = rotary_embedding.rotate_queries_and_keys(&queries, &keys, None);
    }

    #[test]
    fn test_different_seq_dim_rotary_embedding_rotate_queries_and_keys() {
        let queries = q_or_k().i(0).unsqueeze(0).permute(&[1, 0, 2]); // shape: [4, 1, 8]
        let keys = tensor![
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]]
        ].permute(&[1, 0, 2]); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let (actual_queries, actual_keys) = rotary_embedding.rotate_queries_and_keys(&queries, &keys, Some(&-3));

        let expected_queries = tensor![[[ 0.0000000000000000,  1.0030015707015991,  2.0034546852111816,
                3.0023059844970703,  4.0196223258972168,  5.0150079727172852,
                6.0103640556335449,  7.0053806304931641]],
            [[-1.1454386711120605,  1.9249579906463623,  2.5879111289978027,
                4.2811617851257324,  4.9518523216247559,  6.0587716102600098,
                6.9980325698852539,  8.0100736618041992]],
            [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
                5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
                7.9819841384887695,  9.0159816741943359]],
            [[-3.5258200168609619, -3.5313138961791992,  3.0009701251983643,
                7.2068500518798828,  6.7403740882873535,  8.1940803527832031,
                8.9921970367431641,  0.0269895885139704]]];
        let expected_keys = tensor![[[ 2.9853551387786865,  3.9880297183990479,  4.9913783073425293,
                5.9953918457031250,  6.9658288955688477,  7.9760594367980957,
                8.9844808578491211,  0.0000000000000000]],
            [[-1.4402798414230347,  3.2989020347595215,  3.4778468608856201,
                5.3722896575927734,  5.9152107238769531,  7.0490779876708984,
                7.9841032028198242,  9.0045347213745117]],
            [[-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
                4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
                6.9839863777160645,  8.0139846801757812]],
            [[-0.1414657086133957, -0.9914771318435669,  1.0249965190887451,
                3.4583785533905029,  3.8576498031616211,  5.1254072189331055,
                5.9841351509094238,  7.0206656455993652]]];

        assert_tensors_eq(&actual_queries, &expected_queries);
        assert_tensors_eq(&actual_keys, &expected_keys);
    }

    #[test]
    fn test_rotary_embedding_get_scale() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let xpos_rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let seqs = xpos_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        let actual = xpos_rotary_embedding.get_scale(&seqs, None, None);
        let expected = tensor![[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
                1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
                1.0017273426055908, 1.0007686614990234],
            [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
                1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
                1.0008633136749268, 1.0003843307495117],
            [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000],
            [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
                0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
                0.9991374015808105, 0.9996158480644226]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    #[should_panic]
    fn test_rotary_embedding_get_scale_throws_no_xpos() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        // throws because use_xpos needs to be true to call get_scale
        let _actual = rotary_embedding.get_scale(&q_or_k, None, None);
    }

    #[test]
    fn test_different_seq_len_rotary_embedding_get_scale() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let xpos_rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let seqs = xpos_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_none() || xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 0);

        let actual = xpos_rotary_embedding.get_scale(&seqs, Some(&4), None);
        let expected = tensor![[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
                1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
                1.0017273426055908, 1.0007686614990234],
            [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
                1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
                1.0008633136749268, 1.0003843307495117],
            [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000],
            [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
                0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
                0.9991374015808105, 0.9996158480644226]];

        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_some() && xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] >= 4);

        assert_tensors_eq(&actual, &expected);

        let actual = xpos_rotary_embedding.get_scale(&seqs, Some(&8), None);

        // there isn't actually 8 sequence positions in the input, so the cache should not be 8 long
        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_some() && xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] < 8);

        assert_tensors_eq(&actual, &expected);

    }

    #[test]
    fn test_different_offset_rotary_embedding_get_scale() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let xpos_rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let seqs = xpos_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_none() || xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 0);

        let actual = xpos_rotary_embedding.get_scale(&seqs, None, Some(&4));
        let expected = tensor![[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
                1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
                1.0017273426055908, 1.0007686614990234],
            [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
                1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
                1.0008633136749268, 1.0003843307495117],
            [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000],
            [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
                0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
                0.9991374015808105, 0.9996158480644226]];

        // nothing is cached because seq_len not specified
        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_none() || xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 0);

        assert_tensors_eq(&actual, &expected);

        let q_or_k = q_or_k.repeat(&[1, 2, 1]); // shape: [1, 8, 8] basically just a different sequence length

        let seqs = xpos_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        // value of seq_len doesn't matter because all it does is determine whether caching should happen;
        // the actual length of the cached data is of course taken from the data itself before caching
        let actual = xpos_rotary_embedding.get_scale(&seqs, Some(&1203912), Some(&4));
        let expected = tensor![[1.0098352432250977, 1.0060122013092041, 1.0034577846527100,
                1.0015380382537842, 1.0098352432250977, 1.0060122013092041,
                1.0034577846527100, 1.0015380382537842],
            [1.0073673725128174, 1.0045057535171509, 1.0025922060012817,
                1.0011532306671143, 1.0073673725128174, 1.0045057535171509,
                1.0025922060012817, 1.0011532306671143],
            [1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
                1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
                1.0017273426055908, 1.0007686614990234],
            [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
                1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
                1.0008633136749268, 1.0003843307495117],
            [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000],
            [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
                0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
                0.9991374015808105, 0.9996158480644226],
            [0.9951183199882507, 0.9970073699951172, 0.9982755780220032,
                0.9992318749427795, 0.9951183199882507, 0.9970073699951172,
                0.9982755780220032, 0.9992318749427795],
            [0.9926864504814148, 0.9955144524574280, 0.9974144697189331,
                0.9988480806350708, 0.9926864504814148, 0.9955144524574280,
                0.9974144697189331, 0.9988480806350708]];

        // sequence length of 8 is cached
        assert!(xpos_rotary_embedding.varstore.get("cached_scales").is_some() && xpos_rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 8);

        assert_tensors_eq(&actual, &expected);

        // value of seq_len does matter for subsequent calls because it checks against the cache's length
        // the "let freqs = varstore.var_copy("freqs", &freqs_tensor);" line in the init implicitly converts freqs to a Float kind
        // tensor, so we just convert the result of this calculation to double kind to compare with our expected value from python
        let actual = xpos_rotary_embedding.get_scale(&seqs, Some(&8), None).to_kind(Kind::Double);
        let expected = tensor![[1.0098352432250977, 1.0060122013092041, 1.0034577846527100,
                1.0015380382537842, 1.0098352432250977, 1.0060122013092041,
                1.0034577846527100, 1.0015380382537842],
            [1.0073673725128174, 1.0045057535171509, 1.0025922060012817,
                1.0011532306671143, 1.0073673725128174, 1.0045057535171509,
                1.0025922060012817, 1.0011532306671143],
            [1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
                1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
                1.0017273426055908, 1.0007686614990234],
            [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
                1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
                1.0008633136749268, 1.0003843307495117],
            [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 1.0000000000000000],
            [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
                0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
                0.9991374015808105, 0.9996158480644226],
            [0.9951183199882507, 0.9970073699951172, 0.9982755780220032,
                0.9992318749427795, 0.9951183199882507, 0.9970073699951172,
                0.9982755780220032, 0.9992318749427795],
            [0.9926864504814148, 0.9955144524574280, 0.9974144697189331,
                0.9988480806350708, 0.9926864504814148, 0.9955144524574280,
                0.9974144697189331, 0.9988480806350708]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_rotary_embedding_get_axial_freqs() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let xpos_rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let actual_no_xpos = rotary_embedding.get_axial_freqs(&[2, 2]);
        let actual_xpos = xpos_rotary_embedding.get_axial_freqs(&[2, 2]);

        assert_tensors_eq(&actual_no_xpos, &actual_xpos);

        let expected = tensor![[[0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 1.0000000000000000,
                1.0000000000000000, 0.1000000014901161, 0.1000000014901161,
                0.0099999997764826, 0.0099999997764826, 0.0010000000474975,
                0.0010000000474975]],
            [[1.0000000000000000, 1.0000000000000000, 0.1000000014901161,
                0.1000000014901161, 0.0099999997764826, 0.0099999997764826,
                0.0010000000474975, 0.0010000000474975, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000],
            [1.0000000000000000, 1.0000000000000000, 0.1000000014901161,
                0.1000000014901161, 0.0099999997764826, 0.0099999997764826,
                0.0010000000474975, 0.0010000000474975, 1.0000000000000000,
                1.0000000000000000, 0.1000000014901161, 0.1000000014901161,
                0.0099999997764826, 0.0099999997764826, 0.0010000000474975,
                0.0010000000474975]]];

        assert_tensors_eq(&actual_no_xpos, &expected);

        let actual_no_xpos = rotary_embedding.get_axial_freqs(&[2]);

        let expected = expected.i(0).i((.., 8..));

        assert_tensors_eq(&actual_no_xpos, &expected);

        let actual_no_xpos = rotary_embedding.get_axial_freqs(&[1]);

        let expected = expected.i((..1, ..)); // result is not repeated twice along the first dimension

        assert_tensors_eq(&actual_no_xpos, &expected);
    }

    #[test]
    fn test_pixel_rotary_embedding_get_axial_freqs() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let pixel_rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("pixel"),
            None, None, None, None, None, None, None, None, None, None
        );

        let actual = pixel_rotary_embedding.get_axial_freqs(&[2, 2]);

        let expected = tensor![[[ -3.1415927410125732,  -3.1415927410125732,  -7.3303837776184082,
                -7.3303837776184082, -11.5191726684570312, -11.5191726684570312,
                -15.7079639434814453, -15.7079639434814453,  -3.1415927410125732,
                -3.1415927410125732,  -7.3303837776184082,  -7.3303837776184082,
                -11.5191726684570312, -11.5191726684570312, -15.7079639434814453,
                -15.7079639434814453],
            [ -3.1415927410125732,  -3.1415927410125732,  -7.3303837776184082,
                -7.3303837776184082, -11.5191726684570312, -11.5191726684570312,
                -15.7079639434814453, -15.7079639434814453,   3.1415927410125732,
                3.1415927410125732,   7.3303837776184082,   7.3303837776184082,
                11.5191726684570312,  11.5191726684570312,  15.7079639434814453,
                15.7079639434814453]],
            [[  3.1415927410125732,   3.1415927410125732,   7.3303837776184082,
                7.3303837776184082,  11.5191726684570312,  11.5191726684570312,
                15.7079639434814453,  15.7079639434814453,  -3.1415927410125732,
                -3.1415927410125732,  -7.3303837776184082,  -7.3303837776184082,
                -11.5191726684570312, -11.5191726684570312, -15.7079639434814453,
                -15.7079639434814453],
            [  3.1415927410125732,   3.1415927410125732,   7.3303837776184082,
                7.3303837776184082,  11.5191726684570312,  11.5191726684570312,
                15.7079639434814453,  15.7079639434814453,   3.1415927410125732,
                3.1415927410125732,   7.3303837776184082,   7.3303837776184082,
                11.5191726684570312,  11.5191726684570312,  15.7079639434814453,
                15.7079639434814453]]];

        assert_tensors_eq(&actual, &expected);

        let actual = pixel_rotary_embedding.get_axial_freqs(&[2]);

        let expected = expected.i(0).i((.., 8..));

        assert_tensors_eq(&actual, &expected);

        let actual = pixel_rotary_embedding.get_axial_freqs(&[1]);

        let expected = expected.i((..1, ..)); // result is not repeated twice along the first dimension

        assert_tensors_eq(&actual, &expected);
    }
    
    #[test]
    fn test_rotary_embedding_forward() {
        let q_or_k = q_or_k().i(0).unsqueeze(0); // shape: [1, 4, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let seqs = rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(rotary_embedding.varstore.get("cached_scales").is_none() || rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 0);

        // the "let freqs = varstore.var_copy("freqs", &freqs_tensor);" line in the init implicitly converts freqs to a Float kind
        // tensor, so we just convert the result of this calculation to double kind to compare with our expected value from python
        let actual = rotary_embedding.forward(&seqs, None, None).to_kind(Kind::Double);
        let expected = tensor![[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
            0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
            0.0000000000000000e+00, 0.0000000000000000e+00],
        [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
            1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
            1.0000000474974513e-03, 1.0000000474974513e-03],
        [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
            2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
            2.0000000949949026e-03, 2.0000000949949026e-03],
        [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
            3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
            3.0000000260770321e-03, 3.0000000260770321e-03]];

        assert_tensors_eq(&actual, &expected);

        assert!(rotary_embedding.varstore.get("cached_scales").is_none() || rotary_embedding.varstore.get("cached_scales").unwrap().size()[0] == 4);
    }

    #[test]
    fn test_different_seq_len_rotary_embedding_forward() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).repeat(&[1, 2, 1]); // shape: [1, 8, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let seqs = rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(rotary_embedding.varstore.get("cached_freqs").is_none() || rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);

        // again, seq_len just needs to be specified for caching to happen initially
        let actual = rotary_embedding.forward(&seqs, Some(&125125124), None).to_kind(Kind::Double);
        let expected = tensor![[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
                1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
                1.0000000474974513e-03, 1.0000000474974513e-03],
            [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
                2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
                2.0000000949949026e-03, 2.0000000949949026e-03],
            [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
                3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
                3.0000000260770321e-03, 3.0000000260770321e-03],
            [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
                4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
                4.0000001899898052e-03, 4.0000001899898052e-03],
            [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
                5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
                5.0000003539025784e-03, 5.0000003539025784e-03],
            [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
                6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
                6.0000000521540642e-03, 6.0000000521540642e-03],
            [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
                6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
                7.0000002160668373e-03, 7.0000002160668373e-03]];

        assert!(rotary_embedding.varstore.get("cached_freqs").is_some() && rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 8);
        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_different_offset_rotary_embedding_forward() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).repeat(&[1, 2, 1]); // shape: [1, 8, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let seqs = rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(rotary_embedding.varstore.get("cached_freqs").is_none() || rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);

        // offset changes nothing if seq_len is not set
        let actual = rotary_embedding.forward(&seqs, None, Some(&125126125)).to_kind(Kind::Double);
        let expected = tensor![[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
                1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
                1.0000000474974513e-03, 1.0000000474974513e-03],
            [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
                2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
                2.0000000949949026e-03, 2.0000000949949026e-03],
            [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
                3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
                3.0000000260770321e-03, 3.0000000260770321e-03],
            [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
                4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
                4.0000001899898052e-03, 4.0000001899898052e-03],
            [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
                5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
                5.0000003539025784e-03, 5.0000003539025784e-03],
            [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
                6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
                6.0000000521540642e-03, 6.0000000521540642e-03],
            [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
                6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
                7.0000002160668373e-03, 7.0000002160668373e-03]];

        assert!(rotary_embedding.varstore.get("cached_freqs").is_none() || rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);
        assert_tensors_eq(&actual, &expected);

        // specifiying seq_len causes caching to happen
        let actual = rotary_embedding.forward(&seqs, Some(&125126126), Some(&4)).to_kind(Kind::Double);
        let expected = tensor![[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
                1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
                1.0000000474974513e-03, 1.0000000474974513e-03],
            [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
                2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
                2.0000000949949026e-03, 2.0000000949949026e-03],
            [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
                3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
                3.0000000260770321e-03, 3.0000000260770321e-03],
            [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
                4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
                4.0000001899898052e-03, 4.0000001899898052e-03],
            [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
                5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
                5.0000003539025784e-03, 5.0000003539025784e-03],
            [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
                6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
                6.0000000521540642e-03, 6.0000000521540642e-03],
            [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
                6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
                7.0000002160668373e-03, 7.0000002160668373e-03]];

        assert!(rotary_embedding.varstore.get("cached_freqs").is_some() && rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 8);
        assert_tensors_eq(&actual, &expected);

        // now to get cached value seq_len has to be 8
        let actual = rotary_embedding.forward(&seqs, Some(&8), Some(&4)).to_kind(Kind::Double);

        assert_tensors_eq(&actual, &expected);

        let actual = rotary_embedding.forward(&seqs, Some(&6), Some(&2)).to_kind(Kind::Double);
        let expected = tensor![[2.00000000000000000000e+00, 2.00000000000000000000e+00,
                2.00000002980232238770e-01, 2.00000002980232238770e-01,
                1.99999995529651641846e-02, 1.99999995529651641846e-02,
                2.00000009499490261078e-03, 2.00000009499490261078e-03],
            [3.00000000000000000000e+00, 3.00000000000000000000e+00,
                3.00000011920928955078e-01, 3.00000011920928955078e-01,
                2.99999993294477462769e-02, 2.99999993294477462769e-02,
                3.00000002607703208923e-03, 3.00000002607703208923e-03],
            [4.00000000000000000000e+00, 4.00000000000000000000e+00,
                4.00000005960464477539e-01, 4.00000005960464477539e-01,
                3.99999991059303283691e-02, 3.99999991059303283691e-02,
                4.00000018998980522156e-03, 4.00000018998980522156e-03],
            [5.00000000000000000000e+00, 5.00000000000000000000e+00,
                5.00000000000000000000e-01, 5.00000000000000000000e-01,
                4.99999970197677612305e-02, 4.99999970197677612305e-02,
                5.00000035390257835388e-03, 5.00000035390257835388e-03],
            [6.00000000000000000000e+00, 6.00000000000000000000e+00,
                6.00000023841857910156e-01, 6.00000023841857910156e-01,
                5.99999986588954925537e-02, 5.99999986588954925537e-02,
                6.00000005215406417847e-03, 6.00000005215406417847e-03],
            [7.00000000000000000000e+00, 7.00000000000000000000e+00,
                6.99999988079071044922e-01, 6.99999988079071044922e-01,
                7.00000002980232238770e-02, 7.00000002980232238770e-02,
                7.00000021606683731079e-03, 7.00000021606683731079e-03]];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_pixel_rotary_embedding_forward_doesnt_cache() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).repeat(&[1, 2, 1]); // shape: [1, 8, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let pixel_rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("pixel"),
            None, None, None, None, None, None, None, None, None, None
        );

        let seqs = pixel_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(pixel_rotary_embedding.varstore.get("cached_freqs").is_none() || pixel_rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);

        // normally seq_len being specified would cause caching to happen, but since freqs_for is pixel it does not
        let _actual = pixel_rotary_embedding.forward(&seqs, Some(&8), None);

        assert!(pixel_rotary_embedding.varstore.get("cached_freqs").is_none() || pixel_rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);
    }

    #[test]
    fn test_learned_freq_rotary_embedding_forward_doesnt_cache() {
        let q_or_k = q_or_k().i(0).unsqueeze(0).repeat(&[1, 2, 1]); // shape: [1, 8, 8]

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let learned_freq_rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None, None, None, None, None,
            Some(&true),
            None, None, None, None, None, None
        );

        let seqs = learned_freq_rotary_embedding.get_seq_pos(&q_or_k.size()[1], &q_or_k.kind(), &q_or_k.device(), None);

        assert!(learned_freq_rotary_embedding.varstore.get("cached_freqs").is_none() || learned_freq_rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);

        // normally seq_len being specified would cause caching to happen, but since the frequencies are learned caching does not happen
        let _actual = learned_freq_rotary_embedding.forward(&seqs, Some(&8), None);

        assert!(learned_freq_rotary_embedding.varstore.get("cached_freqs").is_none() || learned_freq_rotary_embedding.varstore.get("cached_freqs").unwrap().size()[0] == 0);
    }

    #[test]
    fn test_rotary_embedding_init_freqs_calculation() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();

        // default freqs_for is 'lang'
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.0000000000000000, 0.1000000014901161, 0.0099999997764826, 0.0010000000474975];

        assert_tensors_eq(&actual, &expected);

        let expected = tensor![1., 2., 3., 4.];
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            Some(&expected),
            None, None, None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);

        assert_tensors_eq(&actual, &expected);

        // this allows for normally invalid data states
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM / 2,
            Some(&expected),
            None, None, None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);

        assert_tensors_eq(&actual, &expected);

        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("pixel"),
            None, None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![3.1415927410125732, 7.3303837776184082, 11.5191726684570312, 15.7079639434814453];

        assert_tensors_eq(&actual, &expected);

        // max_freq is only used when freqs_for="pixel"; double the default in this case changes freqs
        // notably its not an actual upper bound for the contents of the calculated freqs tensor
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("pixel"),
            None,
            Some(&20.),
            None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![3.1415927410125732, 12.5663709640502930, 21.9911499023437500, 31.4159278869628906];

        assert_tensors_eq(&actual, &expected);

        // freqs_for = "constant"
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("constant"),
            None, None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.];

        assert_tensors_eq(&actual, &expected);

        // num_freqs only used when freqs_for = "constant"
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None,
            Some("constant"),
            None, None,
            Some(&3),
            None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1., 1., 1.];

        assert_tensors_eq(&actual, &expected);

        // specifying theta (double default to see effects)
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None, None,
            Some(&20000.),
            None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.0000000000000000e+00, 8.4089644253253937e-02, 7.0710680447518826e-03, 5.9460353804752231e-04];

        assert_tensors_eq(&actual, &expected);

        // half theta
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None, None,
            Some(&5000.),
            None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.0000000000000000, 0.1189207136631012, 0.0141421360895038, 0.0016817927826196];

        assert_tensors_eq(&actual, &expected);

        // theta_rescale_factor affects calculation of freqs for 'lang' only
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None, None, None, None, None, None, None, None, None,
            Some(&0.5),
            None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.0000000000000000, 0.1259921044111252, 0.0158740114420652, 0.0020000000949949];

        assert_tensors_eq(&actual, &expected);

        // double dim
        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM * 2,
            None, None, None, None, None, None, None, None, None, None, None, None
        );

        let actual = rotary_embedding.freqs.to_kind(Kind::Double);
        let expected = tensor![1.0000000000000000e+00, 3.1622776389122009e-01, 1.0000000149011612e-01,
                3.1622778624296188e-02, 9.9999997764825821e-03, 3.1622778624296188e-03,
                1.0000000474974513e-03, 3.1622778624296188e-04];

        assert_tensors_eq(&actual, &expected);
    }

    #[test]
    fn test_rotary_embedding_init_default_seq_dim() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        assert_eq!(rotary_embedding.default_seq_dim, -2);

        let rotary_embedding = RotaryEmbedding::new(
            &varstore_root,
            DIM,
            None, None, None, None, None, None, None, None, None, None,
            Some(&true),
            None
        );

        assert_eq!(rotary_embedding.default_seq_dim, -3);
    }

    #[test]
    #[should_panic]
    fn test_rotary_embedding_init_throws_invalid_interpolate_factor1() {
        RotaryEmbedding::new(
            &nn::VarStore::new(Device::Cpu).root(),
            DIM,
            None, None, None, None, None, None, None, None,
            Some(&0.),
            None, None, None
        );
    }

    #[test]
    #[should_panic]
    fn test_rotary_embedding_init_throws_invalid_interpolate_factor2() {
        RotaryEmbedding::new(
            &nn::VarStore::new(Device::Cpu).root(),
            DIM,
            None, None, None, None, None, None, None, None,
            Some(&-1.),
            None, None, None
        );
    }

    #[test]
    fn test_rotary_embedding_init_not_xpos_add_none_scale_attr() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_default(&varstore_root, DIM);

        assert!(rotary_embedding.varstore.get("scale").is_none());
    }

    #[test]
    fn test_rotary_embedding_init_xpos() {
        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM, None);

        assert!(rotary_embedding.varstore.get("scale").is_some());

        let actual = rotary_embedding.varstore.get("scale").unwrap().to_kind(Kind::Double);
        let expected = tensor![0.2857142984867096, 0.4642857015132904, 0.6428571343421936, 0.8214285969734192];

        assert_tensors_eq(&actual, &expected);

        let varstore = nn::VarStore::new(Device::Cpu);
        let varstore_root = varstore.root();
        let rotary_embedding = RotaryEmbedding::new_xpos(&varstore_root, DIM * 2, None);

        assert!(rotary_embedding.varstore.get("scale").is_some());

        let actual = rotary_embedding.varstore.get("scale").unwrap().to_kind(Kind::Double);
        let expected = tensor![0.2857142984867096, 0.3750000000000000, 0.4642857015132904, 0.5535714030265808, 0.6428571343421936, 0.7321428656578064, 0.8214285969734192, 0.9107142686843872];

        assert_tensors_eq(&actual, &expected);
    }
}