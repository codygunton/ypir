/// Direct test of YPIR SimplePIR with the exact parameters used by ypir-cpu-server.
use ypir::params::params_for_scenario_simplepir;
use ypir::server::{DbRowsPadded, YServer};

fn main() {
    let num_items = 4213;
    let item_size_bits = 20480 * 8; // 163840 bits

    println!("Computing params for {} items, {} bits each...", num_items, item_size_bits);
    let params = params_for_scenario_simplepir(num_items, item_size_bits);

    let db_rows = 1usize << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let db_rows_padded = params.db_rows_padded();

    println!(
        "Params: poly_len={}, nu_1={}, instances={}, pt_modulus={}",
        params.poly_len, params.db_dim_1, params.instances, params.pt_modulus
    );
    println!("DB: {} rows x {} cols, padded rows: {}", db_rows, db_cols, db_rows_padded);

    // Create a small DB of zeros
    let db_data = vec![0u16; db_rows * db_cols];

    println!("Creating YServer...");
    let y_server = YServer::<u16>::new(&params, db_data.into_iter(), true, false, true);
    println!("YServer created");

    println!("Offline precomputation...");
    let offline_vals = y_server.perform_offline_precomputation_simplepir(None);
    println!("Offline precomputation done");

    // Create a zero query — MUST be 64-byte aligned for AVX-512
    use spiral_rs::aligned_memory::AlignedMemory64;
    let packed_query_aligned = AlignedMemory64::new(db_rows_padded);
    let packed_query = packed_query_aligned.as_slice();

    // Create dummy expansion params (using the actual YPIR client)
    use spiral_rs::client::Client;
    use spiral_rs::poly::*;
    let mut client = Client::init(&params);
    client.generate_secret_keys();

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use spiral_rs::discrete_gaussian::DiscreteGaussian;
    use spiral_rs::gadget::build_gadget;

    let static_seed_2: [u8; 32] = [
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ];

    let sk_reg = client.get_sk_reg().clone();
    let num_exp = params.poly_len_log2;
    let m_exp = params.t_exp_left;
    let g_exp = build_gadget(&params, 1, m_exp);
    let g_exp_ntt = g_exp.ntt();
    let dg = DiscreteGaussian::init(params.noise_width);

    let mut rng = ChaCha20Rng::from_entropy();
    let mut rng_pub = ChaCha20Rng::from_seed(static_seed_2);

    let mut pack_pub_params = Vec::new();
    for i in 0..num_exp {
        let t = (params.poly_len / (1 << i)) + 1;
        let tau_sk_reg = spiral_rs::poly::automorph_alloc(&sk_reg, t);
        let prod = &tau_sk_reg.ntt() * &g_exp_ntt;

        // Fresh public key sample
        let mut p = PolyMatrixNTT::zero(&params, 2, m_exp);
        for j in 0..m_exp {
            let a = PolyMatrixRaw::random_rng(&params, 1, 1, &mut rng_pub);
            let e = PolyMatrixRaw::noise(&params, 1, 1, &dg, &mut rng);
            let b = &sk_reg.ntt() * &a.ntt();
            let b = &e.ntt() + &b;
            let mut sample = PolyMatrixNTT::zero(&params, 2, 1);
            sample.copy_into(&(-&a).ntt(), 0, 0);
            sample.copy_into(&b, 1, 0);
            p.copy_into(&sample, 0, j);
        }
        let w = &p + &prod.pad_top(1);
        pack_pub_params.push(w);
    }

    // Extract row-1 and condense
    use ypir::packing::condense_matrix;
    let mut pack_pub_params_row_1s = Vec::new();
    for pp in &pack_pub_params {
        let row_1 = pp.submatrix(1, 0, 1, pp.cols);
        let condensed = condense_matrix(&params, &row_1);
        pack_pub_params_row_1s.push(condensed);
    }

    println!("Running online computation...");
    let responses = y_server.perform_online_computation_simplepir(
        &packed_query,
        &offline_vals,
        &[pack_pub_params_row_1s.as_slice()],
        None,
    );
    println!(
        "Done! Got {} response ciphertexts, total {} bytes",
        responses.len(),
        responses.iter().map(|r| r.len()).sum::<usize>()
    );
}
