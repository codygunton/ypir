//! Persistable form of the expensive SimplePIR offline "tail".
//!
//! `perform_offline_precomputation_simplepir_from_hint` spends nearly all of its time in
//! `prep_pack_many_lwes` and `precompute_pack`. Both are deterministic functions of
//! `(hint_0, params)`, so their outputs can be written to disk once at build time and read back
//! at server start, removing the recompute from the critical path entirely.
//!
//! # Why this is sound
//!
//! `precompute_pack` takes `fake_pack_pub_params`, which *is* drawn from entropy, so it is worth
//! spelling out why the values persisted here are nevertheless deterministic.
//!
//! `generate_fake_pack_pub_params` calls `raw_generate_expansion_params` with a zero secret key.
//! `get_reg_sample` puts `-a` in row 0, drawn from `ChaCha20Rng::from_seed(STATIC_SEED_2)` and so
//! fixed, and `b = e + sk*a = e` in row 1, drawn from `from_entropy()`. The entropy therefore lives
//! in **row 1 only**.
//!
//! Inside `precompute_pack` the gadget-inverse chain reads row 0 alone: `from_ntt_scratch` puts
//! row 0 into the 1x1 `ct_raw`, and `automorph` / `gadget_invert_rdim` act on that. So every
//! entry of `precomp.1` is a function of row 0 and is fully deterministic. Likewise
//! `multiply_no_reduce(&mut w_times_ginv_ct, &pub_param, ...)` yields a 2x1 whose row 0 comes from
//! `pub_param` row 0 (deterministic) and whose row 1 comes from `pub_param` row 1 (entropy). That
//! row-1 taint accumulates into `working_set[0]` = `precomp.0`, but `pack_using_precomp_vals` does
//! `res.get_poly_mut(1, 0).copy_from_slice(resulting_row_1)` and overwrites it wholesale.
//!
//! So: `precomp.0` row 1 is the only entropy-tainted value, and it is discarded online. We zero it
//! on extraction rather than storing garbage, which also makes the serialized artifact
//! byte-reproducible across builds. `simplepir_precomp_ignores_res_row_1` pins the invariant.
//!
//! `precomp.2` is produced by `generate_automorph_tables_brute_force`, which draws a random
//! polynomial but only uses it to *discover* a fixed NTT permutation; the table it returns is a
//! deterministic function of `params`.
//!
//! # What is not stored
//!
//! `prepacked_lwe` is needed to *build* `precomp`, but `pack_many_lwes` reads it only in two
//! `assert_eq!` shape checks -- no element of it reaches the online result. We therefore neither
//! store nor rebuild it: a restored server leaves the field empty, which `pack_many_lwes` accepts.
//! Materializing it as zeros instead would cost ~490 ms and 1.5 GiB of resident memory per cell
//! for values nothing reads. `simplepir_online_ignores_prepacked_lwe_contents` pins the invariant
//! and would fail loudly if a future refactor started reading the contents.
//!
//! # Encoding
//!
//! Little-endian throughout. A matrix is stored as `rows`, `cols`, `polys_stored`,
//! `words_per_poly`, then `polys_stored * words_per_poly` u64s: the leading `words_per_poly` words
//! of each of the leading `polys_stored` polynomials, with everything else known to be zero. Both
//! counts are computed from the data, so the trimming is lossless by construction rather than by
//! assumption. It matters because `condense_matrix` packs both CRT limbs into the low `poly_len`
//! words of every polynomial, leaving the upper half zero, which halves `precomp.1` -- the bulk of
//! the artifact.

use std::fmt;

use spiral_rs::params::Params;
use spiral_rs::poly::*;

use crate::server::{OfflinePrecomputedValues, Precomp};

const MAGIC: [u8; 8] = *b"YPIRTAIL";
const VERSION: u32 = 1;

/// Returned instead of panicking so callers can fall back to recomputing from the hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailDecodeError(String);

impl fmt::Display for TailDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed offline tail: {}", self.0)
    }
}

impl std::error::Error for TailDecodeError {}

fn err<T>(msg: impl Into<String>) -> Result<T, TailDecodeError> {
    Err(TailDecodeError(msg.into()))
}

/// The params a tail was built against, checked on decode so an artifact from a different
/// deployment is rejected rather than silently producing wrong answers.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TailHeader {
    poly_len: u32,
    crt_count: u32,
    poly_len_log2: u32,
    t_exp_left: u32,
    instances: u32,
    modulus: u64,
    /// Fingerprint of the `hint_0` this tail was built from. The params alone cannot catch a tail
    /// paired with a different hint of the same shape -- `db_dim_1` does not appear anywhere in
    /// the tail -- and that pairing would silently answer queries wrongly.
    hint_checksum: u64,
}

impl TailHeader {
    fn for_params(params: &Params, hint_0: &[u64]) -> Self {
        Self {
            poly_len: params.poly_len as u32,
            crt_count: params.crt_count as u32,
            poly_len_log2: params.poly_len_log2 as u32,
            t_exp_left: params.t_exp_left as u32,
            instances: params.instances as u32,
            modulus: params.modulus,
            hint_checksum: hint_checksum(hint_0),
        }
    }

    fn check_params(&self, params: &Params) -> Result<(), TailDecodeError> {
        let expected = [
            ("poly_len", self.poly_len as usize, params.poly_len),
            ("crt_count", self.crt_count as usize, params.crt_count),
            (
                "poly_len_log2",
                self.poly_len_log2 as usize,
                params.poly_len_log2,
            ),
            ("t_exp_left", self.t_exp_left as usize, params.t_exp_left),
            ("instances", self.instances as usize, params.instances),
        ];
        for (name, got, want) in expected {
            if got != want {
                return err(format!("{} is {}, expected {}", name, got, want));
            }
        }
        if self.modulus != params.modulus {
            return err(format!(
                "modulus is {}, expected {}",
                self.modulus, params.modulus
            ));
        }
        Ok(())
    }

    fn num_words(&self) -> usize {
        self.poly_len as usize * self.crt_count as usize
    }
}

/// A `PolyMatrixNTT` reduced to plain owned coefficients, with all-zero tails dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OwnedPolyMatrix {
    rows: usize,
    cols: usize,
    polys_stored: usize,
    words_per_poly: usize,
    data: Vec<u64>,
}

impl OwnedPolyMatrix {
    /// `data` is the full `rows * cols * num_words` coefficient slice.
    fn from_coeffs(rows: usize, cols: usize, num_words: usize, data: &[u64]) -> Self {
        let num_polys = rows * cols;
        assert_eq!(data.len(), num_polys * num_words);

        let mut polys_stored = 0;
        let mut words_per_poly = 0;
        for p in 0..num_polys {
            let poly = &data[p * num_words..][..num_words];
            if let Some(last) = poly.iter().rposition(|&x| x != 0) {
                polys_stored = p + 1;
                words_per_poly = words_per_poly.max(last + 1);
            }
        }

        let mut trimmed = Vec::with_capacity(polys_stored * words_per_poly);
        for p in 0..polys_stored {
            trimmed.extend_from_slice(&data[p * num_words..][..words_per_poly]);
        }

        Self {
            rows,
            cols,
            polys_stored,
            words_per_poly,
            data: trimmed,
        }
    }

    fn to_ntt<'a>(&self, params: &'a Params, num_words: usize) -> PolyMatrixNTT<'a> {
        let mut res = PolyMatrixNTT::zero(params, self.rows, self.cols);
        let slc = res.as_mut_slice();
        for p in 0..self.polys_stored {
            slc[p * num_words..][..self.words_per_poly]
                .copy_from_slice(&self.data[p * self.words_per_poly..][..self.words_per_poly]);
        }
        res
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        push_u32(out, self.rows as u32);
        push_u32(out, self.cols as u32);
        push_u32(out, self.polys_stored as u32);
        push_u32(out, self.words_per_poly as u32);
        for &word in &self.data {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }

    fn read_from(r: &mut Reader, num_words: usize) -> Result<Self, TailDecodeError> {
        let rows = r.u32()? as usize;
        let cols = r.u32()? as usize;
        let polys_stored = r.u32()? as usize;
        let words_per_poly = r.u32()? as usize;

        if rows == 0 || cols == 0 || rows > 64 || cols > 64 {
            return err(format!("implausible matrix shape {}x{}", rows, cols));
        }
        if polys_stored > rows * cols {
            return err(format!(
                "{} stored polys exceeds the {} in a {}x{} matrix",
                polys_stored,
                rows * cols,
                rows,
                cols
            ));
        }
        if words_per_poly > num_words {
            return err(format!(
                "{} words per poly exceeds the {} implied by params",
                words_per_poly, num_words
            ));
        }

        Ok(Self {
            rows,
            cols,
            polys_stored,
            words_per_poly,
            data: r.u64s(polys_stored * words_per_poly)?,
        })
    }
}

/// The deterministic, expensive part of the SimplePIR offline precomputation, owned and
/// serializable. Pair it with the separately persisted `hint_0` to rebuild an
/// [`OfflinePrecomputedValues`] with no packing work at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineTail {
    header: TailHeader,
    /// `generate_automorph_tables_brute_force` output. Identical for every `precomp` entry, so it
    /// is stored once and cloned back out on rebuild.
    tables: Vec<Vec<usize>>,
    precomp_res: Vec<OwnedPolyMatrix>,
    precomp_vals: Vec<Vec<OwnedPolyMatrix>>,
}

impl OfflineTail {
    /// Serialized length in bytes, without building the buffer.
    pub fn encoded_len(&self) -> usize {
        let matrix_len = |m: &OwnedPolyMatrix| 16 + m.data.len() * 8;
        let mut n = MAGIC.len() + 4 + 4 * 5 + 8 + 8;
        n += 4 + self.tables.iter().map(|t| 4 + t.len() * 4).sum::<usize>();
        n += 4;
        for (res, vals) in self.precomp_res.iter().zip(self.precomp_vals.iter()) {
            n += matrix_len(res) + 4 + vals.iter().map(matrix_len).sum::<usize>();
        }
        n
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&MAGIC);
        push_u32(&mut out, VERSION);

        let h = &self.header;
        for v in [
            h.poly_len,
            h.crt_count,
            h.poly_len_log2,
            h.t_exp_left,
            h.instances,
        ] {
            push_u32(&mut out, v);
        }
        out.extend_from_slice(&h.modulus.to_le_bytes());
        out.extend_from_slice(&h.hint_checksum.to_le_bytes());

        push_u32(&mut out, self.tables.len() as u32);
        for table in &self.tables {
            push_u32(&mut out, table.len() as u32);
            for &entry in table {
                push_u32(&mut out, entry as u32);
            }
        }

        push_u32(&mut out, self.precomp_res.len() as u32);
        for (res, vals) in self.precomp_res.iter().zip(self.precomp_vals.iter()) {
            res.write_to(&mut out);
            push_u32(&mut out, vals.len() as u32);
            for val in vals {
                val.write_to(&mut out);
            }
        }

        debug_assert_eq!(out.len(), self.encoded_len());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TailDecodeError> {
        let mut r = Reader::new(bytes);

        if r.take(MAGIC.len())? != MAGIC {
            return err("bad magic");
        }
        let version = r.u32()?;
        if version != VERSION {
            return err(format!("version {}, expected {}", version, VERSION));
        }

        let header = TailHeader {
            poly_len: r.u32()?,
            crt_count: r.u32()?,
            poly_len_log2: r.u32()?,
            t_exp_left: r.u32()?,
            instances: r.u32()?,
            modulus: r.u64()?,
            hint_checksum: r.u64()?,
        };
        if header.poly_len == 0 || header.crt_count == 0 {
            return err("zero poly_len or crt_count");
        }
        let num_words = header.num_words();

        let num_tables = r.u32()? as usize;
        let mut tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            let len = r.u32()? as usize;
            let mut table = Vec::with_capacity(len);
            for _ in 0..len {
                let entry = r.u32()? as usize;
                if entry >= header.poly_len as usize {
                    return err(format!("automorph table entry {} out of range", entry));
                }
                table.push(entry);
            }
            tables.push(table);
        }

        let num_precomp = r.u32()? as usize;
        let mut precomp_res = Vec::with_capacity(num_precomp);
        let mut precomp_vals = Vec::with_capacity(num_precomp);
        for _ in 0..num_precomp {
            precomp_res.push(OwnedPolyMatrix::read_from(&mut r, num_words)?);
            let num_vals = r.u32()? as usize;
            let mut vals = Vec::with_capacity(num_vals);
            for _ in 0..num_vals {
                vals.push(OwnedPolyMatrix::read_from(&mut r, num_words)?);
            }
            precomp_vals.push(vals);
        }

        if !r.is_empty() {
            return err(format!("{} trailing bytes", r.remaining()));
        }

        Ok(Self {
            header,
            tables,
            precomp_res,
            precomp_vals,
        })
    }

    /// Checks the tail against `params` and against the shapes the online path requires.
    fn validate(&self, params: &Params, hint_0: &[u64]) -> Result<(), TailDecodeError> {
        self.header.check_params(params)?;

        let checksum = hint_checksum(hint_0);
        if checksum != self.header.hint_checksum {
            return err(format!(
                "tail was built for a different hint_0 (checksum {:#x}, got {:#x})",
                self.header.hint_checksum, checksum
            ));
        }

        // db_cols / poly_len, where db_cols = instances * poly_len
        let num_rlwe_outputs = params.instances;
        if self.precomp_res.len() != num_rlwe_outputs {
            return err(format!(
                "precomp has {} entries, expected {}",
                self.precomp_res.len(),
                num_rlwe_outputs
            ));
        }
        if self.tables.len() != params.poly_len_log2 {
            return err(format!(
                "{} automorph tables, expected {}",
                self.tables.len(),
                params.poly_len_log2
            ));
        }
        for table in &self.tables {
            if table.len() != params.poly_len {
                return err(format!(
                    "automorph table has {} entries, expected {}",
                    table.len(),
                    params.poly_len
                ));
            }
        }

        // `pack_using_precomp_vals` walks `precomp_vals` in lockstep with its own butterfly and
        // asserts it consumed exactly all of them, so a wrong count must be caught here.
        let expected_vals: usize = (1..=params.poly_len_log2)
            .map(|cur_ell| 1 << (params.poly_len_log2 - cur_ell))
            .sum();
        for vals in &self.precomp_vals {
            if vals.len() != expected_vals {
                return err(format!(
                    "precomp entry has {} vals, expected {}",
                    vals.len(),
                    expected_vals
                ));
            }
            for val in vals {
                if (val.rows, val.cols) != (params.t_exp_left, 1) {
                    return err(format!(
                        "precomp val is {}x{}, expected {}x1",
                        val.rows, val.cols, params.t_exp_left
                    ));
                }
            }
        }
        for res in &self.precomp_res {
            if (res.rows, res.cols) != (2, 1) {
                return err(format!(
                    "precomp res is {}x{}, expected 2x1",
                    res.rows, res.cols
                ));
            }
        }

        Ok(())
    }

    /// Rebuilds `precomp`. This is the whole cost of a restore: a copy of the stored coefficients
    /// into 64-byte-aligned `AlignedMemory64` allocations, which `PolyMatrixNTT` requires and
    /// which no serialization format can avoid.
    fn rebuild<'a>(&self, params: &'a Params) -> Precomp<'a> {
        let num_words = self.header.num_words();
        self
            .precomp_res
            .iter()
            .zip(self.precomp_vals.iter())
            .map(|(res, vals)| {
                (
                    res.to_ntt(params, num_words),
                    vals.iter().map(|v| v.to_ntt(params, num_words)).collect(),
                    self.tables.clone(),
                )
            })
            .collect()
    }
}

impl<'a> OfflinePrecomputedValues<'a> {
    /// Extracts the persistable tail. Panics if called on values that did not come from the
    /// SimplePIR path (`prepacked_lwe` and `precomp` must be populated).
    pub fn offline_tail(&self) -> OfflineTail {
        assert!(
            !self.precomp.is_empty(),
            "offline_tail requires SimplePIR offline values"
        );

        let params = self.precomp[0].0.params;
        let header = TailHeader::for_params(params, &self.hint_0);
        let num_words = header.num_words();

        let mut precomp_res = Vec::with_capacity(self.precomp.len());
        let mut precomp_vals = Vec::with_capacity(self.precomp.len());
        for (res, vals, _) in self.precomp.iter() {
            // Row 1 is the one entropy-tainted value in the whole tail, and
            // `pack_using_precomp_vals` overwrites it before reading it. Zero it so the artifact
            // is reproducible.
            let mut coeffs = res.as_slice().to_vec();
            coeffs[num_words..].fill(0);
            precomp_res.push(OwnedPolyMatrix::from_coeffs(
                res.rows, res.cols, num_words, &coeffs,
            ));

            precomp_vals.push(
                vals.iter()
                    .map(|v| OwnedPolyMatrix::from_coeffs(v.rows, v.cols, num_words, v.as_slice()))
                    .collect(),
            );
        }

        OfflineTail {
            header,
            tables: self.precomp[0].2.clone(),
            precomp_res,
            precomp_vals,
        }
    }
}

/// Assembles offline values from a persisted hint and tail, doing no packing work.
///
/// Kept free of `YServer` so it does not need a database to run.
pub fn offline_values_from_parts<'a>(
    params: &'a Params,
    hint_0: Vec<u64>,
    tail: &OfflineTail,
) -> Result<OfflinePrecomputedValues<'a>, TailDecodeError> {
    let db_cols = params.instances * params.poly_len;
    if hint_0.len() != params.poly_len * db_cols {
        return err(format!(
            "hint_0 has {} entries, expected {}",
            hint_0.len(),
            params.poly_len * db_cols
        ));
    }
    tail.validate(params, &hint_0)?;

    Ok(OfflinePrecomputedValues {
        hint_0,
        hint_1: vec![],
        pseudorandom_query_1: vec![],
        y_constants: crate::server::generate_y_constants(params),
        smaller_server: None,
        // Left empty deliberately; `pack_many_lwes` reads this only for shape checks it skips when
        // it is empty. See the module docs.
        prepacked_lwe: vec![],
        // Only `precompute_pack` consumes these, and we just skipped it. The SimplePIR online path
        // never touches the field.
        fake_pack_pub_params: vec![],
        precomp: tail.rebuild(params),
    })
}

/// Cheap 64-bit fingerprint of `hint_0`, used only to reject a tail paired with the wrong hint.
/// Eight independent FNV lanes so the multiplies pipeline; ~400 MB of hint costs a few ms, which
/// is the one unavoidable pass we accept to keep a mispaired artifact from answering silently.
fn hint_checksum(hint_0: &[u64]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    let mut lanes = [0xcbf29ce484222325u64; 8];
    let mut chunks = hint_0.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, &word) in lanes.iter_mut().zip(chunk) {
            *lane = (*lane ^ word).wrapping_mul(PRIME);
        }
    }
    let mut acc = hint_0.len() as u64;
    for &word in chunks.remainder() {
        acc = (acc ^ word).wrapping_mul(PRIME);
    }
    for lane in lanes {
        acc = (acc ^ lane).wrapping_mul(PRIME);
    }
    acc
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], TailDecodeError> {
        if self.remaining() < n {
            return err(format!(
                "truncated: wanted {} bytes at offset {}, {} left",
                n,
                self.pos,
                self.remaining()
            ));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, TailDecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, TailDecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn u64s(&mut self, n: usize) -> Result<Vec<u64>, TailDecodeError> {
        // `n` comes from the artifact, so bound it before reserving.
        if n > self.remaining() / 8 {
            return err(format!(
                "truncated: wanted {} u64s at offset {}, {} bytes left",
                n,
                self.pos,
                self.remaining()
            ));
        }
        Ok(self
            .take(n * 8)?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}
