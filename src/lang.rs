//! Port of Laya `laya/lang.py`: decides whether the English checkpoint can read a state.
use serde_json::Value;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    ("cyrillic", &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    ("arabic", &[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    ("hangul", &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
    ("kana", &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
    ("han", &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
];

const STOP: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that", "this", "it", "you",
            "have", "has", "not", "but", "on", "at", "be", "as", "from", "will", "can", "would", "there", "their",
            "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur", "pas", "plus", "nous",
            "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für", "auf", "von", "zu",
            "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como", "pero", "son", "está",
            "este", "esta", "todo", "más", "muy", "hay", "sus",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da", "dos", "das", "mas",
            "são", "está", "este", "esta", "muito", "pelo", "pela",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono", "questo", "questa",
            "anche", "come", "più", "nella", "alla",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan", "door", "maar", "ook",
            "worden", "deze", "naar", "wordt",
        ],
    ),
];

const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğı";

/// Python `str.isalpha`: general category L*.
fn is_alpha(c: char) -> bool {
    c.general_category_group() == GeneralCategoryGroup::Letter
}

/// Python `[^\W\d_]` (re.UNICODE): alphanumeric minus decimal digits minus underscore.
fn is_word_char(c: char) -> bool {
    is_alpha(c) || matches!(c.general_category(), GeneralCategory::LetterNumber | GeneralCategory::OtherNumber)
}

fn collect(v: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Object(m) => m.values().for_each(|x| collect(x, depth + 1, out)),
        Value::Array(a) => a.iter().for_each(|x| collect(x, depth + 1, out)),
        _ => {}
    }
}

/// String leaves joined by spaces, first 4000 chars (keys ignored).
pub fn state_text(state: &Value) -> String {
    let mut parts = Vec::new();
    collect(state, 0, &mut parts);
    parts.join(" ").chars().take(4000).collect()
}

pub fn detect_script(text: &str) -> &'static str {
    // Insertion-ordered counts; "latin" is appended last, ties go to the first maximum (Python `max`).
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut latin = 0;
    for ch in text.chars().filter(|&c| is_alpha(c)) {
        let cp = ch as u32;
        if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
            latin += 1;
            continue;
        }
        if let Some((name, _)) = SCRIPT_RANGES.iter().find(|(_, rs)| rs.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
        {
            match counts.iter_mut().find(|(n, _)| n == name) {
                Some(e) => e.1 += 1,
                None => counts.push((name, 1)),
            }
        }
    }
    counts.push(("latin", latin));
    if counts.iter().all(|(_, n)| *n == 0) {
        return "unknown";
    }
    let mut best = counts[0];
    for &c in &counts[1..] {
        if c.1 > best.1 {
            best = c;
        }
    }
    best.0
}

pub fn guess_latin_language(text: &str) -> Option<&'static str> {
    let words: Vec<String> =
        text.split(|c: char| !is_word_char(c)).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect();
    if words.len() < 4 {
        return None;
    }
    let score = |sw: &[&str]| words.iter().filter(|w| sw.contains(&w.as_str())).count();
    let lowered = text.to_lowercase();
    let n = lowered.chars().count();
    let diac = lowered.chars().filter(|c| NON_EN_DIACRITICS.contains(*c)).count();
    let diac_rate = diac as f64 / n.max(1) as f64;
    let en = score(STOP[0].1);
    let (mut best_lg, mut best) = (None, 0);
    for (lg, sw) in &STOP[1..] {
        let s = score(sw);
        if best_lg.is_none() || s > best {
            best_lg = Some(*lg);
            best = s;
        }
    }
    let en_or_none = if en > 0 { Some("en") } else { None };
    if best == 0 && diac_rate < 0.02 {
        return en_or_none;
    }
    if best >= 2.max(en + 2) {
        return best_lg;
    }
    if diac_rate >= 0.04 && best >= en {
        return best_lg;
    }
    en_or_none
}

pub fn is_english(state: &Value) -> bool {
    let text = state_text(state);
    match detect_script(&text) {
        "unknown" => true,
        "latin" => matches!(guess_latin_language(&text), None | Some("en")),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scripts() {
        for (text, want) in [
            ("The customer was charged twice and wants a refund.", "latin"),
            ("Հայերեն", "armenian"),
            ("ՀԱՅԵՐԵՆ", "armenian"),
            ("։֊", "unknown"),
            ("Le client a été facturé deux fois et demande un remboursement.", "latin"),
            ("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
            ("お客様は二重に請求されたため返金を希望しています。", "kana"),
            ("客户被重复扣款要求退款", "han"),
            ("고객이 두 번 청구되어 환불을 원합니다", "hangul"),
            ("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
            ("வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
            ("С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
            ("ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
            ("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
            ("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
            ("", "unknown"),
            ("12345 6789", "unknown"),
        ] {
            assert_eq!(detect_script(text), want, "{text}");
        }
    }

    #[test]
    fn english_or_not() {
        for (text, want) in [
            ("Please refund the duplicate charge on invoice 4411 today.", true),
            ("Հայերեն", false),
            ("refund me", true),
            ("ग्राहक से दो बार शुल्क लिया गया", false),
            ("お客様は二重に請求されました", false),
            ("С клиента дважды сняли деньги", false),
            (
                "Le client a été facturé deux fois et il demande un remboursement pour la facture qui a été payée le mois dernier avec la carte de crédit",
                false,
            ),
            (
                "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
                false,
            ),
        ] {
            assert_eq!(is_english(&json!(text)), want, "{text}");
        }
    }

    #[test]
    fn latin_language() {
        for (text, want) in [
            ("The customer was charged twice and wants a refund for this invoice", Some("en")),
            ("Le client a ete facture deux fois et il demande un remboursement pour la facture", Some("fr")),
            ("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", Some("de")),
            ("El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", Some("es")),
            ("refund", None),
            (
                "Please refund the duplicate charge on invoice 4411 today because we have been waiting for three days and nobody has replied to us",
                Some("en"),
            ),
        ] {
            assert_eq!(guess_latin_language(text), want, "{text}");
        }
    }

    #[test]
    fn state_flattening_ignores_keys() {
        assert!(state_text(&json!({"body": "charged twice", "n": 3})).contains("charged twice"));
        assert!(state_text(&json!({"a": {"b": ["deep"]}})).contains("deep"));
        assert_eq!(state_text(&Value::Null), "");
        assert!(!is_english(&json!({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})));
    }

    #[test]
    fn devanagari_vowel_sign_is_not_a_letter() {
        // U+093F is Mc (not isalpha in Python) although Rust's char::is_alphabetic says true.
        assert!(!is_alpha('\u{093F}'));
    }
}
