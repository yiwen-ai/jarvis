use lingua::LanguageDetectorBuilder;

// pub use lingua::IsoCode639_1;
// pub use lingua::Language;
pub use isolang::Language;
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
            Some(lang) => match Language::from_name(lang.iso_code_639_3().to_string().as_str()) {
                Some(lang) => lang,
                None => Language::default(),
            },
            None => Language::default(),
        }
    }
}
