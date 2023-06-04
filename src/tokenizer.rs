use tiktoken_rs::cl100k_base_singleton;

pub fn tokens_len(s: &str) -> usize {
    let bpe = cl100k_base_singleton();
    let tokens = bpe.lock().encode_with_special_tokens(s);
    tokens.len()
}
