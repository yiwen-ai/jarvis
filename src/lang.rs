pub use isolang::Language;
use lingua::LanguageDetectorBuilder;
use std::str::FromStr;

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

    pub fn detect(&self, text: &str) -> Option<lingua::Language> {
        self.detector.detect_language_of(text)
    }

    pub fn detect_lang(&self, text: &str) -> Language {
        match self.detect(text) {
            Some(lang) => match Language::from_str(lang.iso_code_639_3().to_string().as_str()) {
                Ok(lang) => lang,
                Err(_) => Language::default(),
            },
            None => Language::default(),
        }
    }
}
