use lingua::LanguageDetectorBuilder;

pub use lingua::IsoCode639_1;
pub use lingua::Language;
pub struct LanguageDetector {
    detector: lingua::LanguageDetector,
}

impl LanguageDetector {
    pub fn new() -> Self {
        Self {
            detector: LanguageDetectorBuilder::from_all_languages()
                .with_preloaded_language_models()
                .build(),
        }
    }

    pub fn new_dev() -> Self {
        let langs = vec![Language::English, Language::Chinese, Language::Japanese];
        Self {
            detector: LanguageDetectorBuilder::from_languages(&langs)
                .with_preloaded_language_models()
                .build(),
        }
    }

    pub fn detect(&self, text: &str) -> Option<Language> {
        self.detector.detect_language_of(text)
    }

    pub fn detect_lang(&self, text: &str) -> String {
        match self.detect(text) {
            Some(lang) => normalize_lang(lang.to_string().as_str()),
            None => "".to_string(),
        }
    }
}

pub fn normalize_lang(lang: &str) -> String {
    if lang.is_empty() {
        return lang.to_string();
    }
    let mut rt = String::with_capacity(lang.len());
    let mut chars = lang.chars();
    rt.push(chars.next().unwrap().to_ascii_uppercase());
    for c in chars {
        rt.push(c.to_ascii_lowercase());
    }
    rt
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case<'a> {
        input: &'a str,
        output: &'a str,
    }

    #[test]
    fn normalize_lang_works() {
        let test_cases = vec![
            Case {
                input: "",
                output: "",
            },
            Case {
                input: "english",
                output: "English",
            },
            Case {
                input: "chinese",
                output: "Chinese",
            },
        ];

        for case in test_cases {
            assert_eq!(normalize_lang(case.input), case.output);
        }
    }
}
