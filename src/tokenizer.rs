use tiktoken_rs::cl100k_base_singleton;

pub fn tokens_len(s: &str) -> usize {
    let bpe = cl100k_base_singleton();
    let tokens = bpe.lock().encode_with_special_tokens(s);
    tokens.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_len_works() {
        println!("tokens_len: {}", tokens_len("RFC 8949 introduces the Concise Binary Object Representation (CBOR), a data format designed for small code and message size, extensibility, and streaming capabilities. It provides guidelines for creating CBOR-based protocols and discusses the tagging of items in CBOR. The document also highlights security considerations and the need for specifying the data model used when working with CBOR data. Additionally, it mentions the potential security issues that can arise when converting CBOR data to other formats, such as JSON. The appendix includes examples of encoded CBOR data items and pseudocode for checking the well-formedness of CBOR data."))
    }
}
