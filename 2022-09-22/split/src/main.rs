use std::fs;

struct Trie {
    is_word: Option<u16>,
    children: Vec<Trie>,
}

impl Trie {
    fn new() -> Self {
        Trie {
            is_word: None,
            children: vec![],
        }
    }

    fn insert(&mut self, word: &[u8], value: u16) {
        if word.is_empty() {
            self.is_word = Some(value);
        } else {
            let idx = word[0] as usize;
            if self.children.len() <= idx {
                self.children.resize_with(idx+1, Trie::new);
            }
            self.children[idx].insert(&word[1..], value);
        }
    }

    fn scan(&self, word: &[u8]) -> Option<(usize,u16)> {
        let result = if self.is_word.is_some() {
            Some((0,self.is_word.unwrap()))
        } else {
            None
        };
        if word.is_empty() {
            result
        } else {
            let idx = word[0] as usize;
            if idx < self.children.len() {
                if let Some((r,v)) = self.children[idx].scan(&word[1..]) {
                    Some((r+1,v))
                } else {
                    result
                }
            } else {
                result
            }
        }
    }
}

fn canonicalize(mut word: &[u8]) -> (Vec<u8>, bool, bool) {
    if word.len() <= 1 {
        return (word.to_vec(), false, false);
    }
    let mut space = false;
    let mut capital = false;
    let mut result = vec![];
    if word[0] == b' ' {
        word = &word[1..];
        space = true;
    }
    let mut first = true;
    for ch in word {
        if first && *ch >= b'A' && *ch <= b'Z' {
            result.push(*ch + 32);
            capital = true;
        } else {
            result.push(*ch);
        }
        first = false;
    }
    (result, space, capital)
}


fn main() {
    println!("Building trie");
    let words:Vec<_> = fs::read_to_string("../data/vocab_enwik9.txt").unwrap().split('\n').map(|w|w.to_owned()).take(4096-256).collect();
    let mut trie = Trie::new();
    for b in 0..=255 {
        trie.insert(&[b], b as u16);
    }
    for (i,word) in words.iter().enumerate() {
        if !word.is_empty() {
            trie.insert(word.as_bytes(), i as u16 + 256);
        }
    }
    println!("Reading text");
    let text = fs::read("../data/stripped_enwik9.txt").unwrap();
    println!("Tokenizing");
    let mut i = 0;
    let mut j = 1000000;
    let mut output = vec![];
    loop {
        if i >= text.len() {
            break;
        }
        let (word, space, capital) = canonicalize(&text[i..(i+32).min(text.len())]);
        let (len,v) = trie.scan(&word).unwrap();
        let value = v * 4 + (if space {1} else {0}) + (if capital {2} else {0});
        output.extend_from_slice(&value.to_le_bytes());
        i += len;
        if space {
            i += 1;
        }
        while i > j {
            println!("{j}");
            j += 1000000;
        }
    }
    fs::write("../data/tokens_enwik9", output).unwrap();
}

/*
fn slow_main() {
    println!("Building regex");
    let mut pattern = String::new();
    let mut words:Vec<_> = fs::read_to_string("../data/vocab_enwik9.txt").unwrap().split('\n').map(|w|w.to_owned()).take(65000).collect();
    words.sort_by_key(|w| -(w.len() as i64));
    for word in &words {
        pattern.push_str(word);
        pattern.push_str("|");
    }
    pattern.push_str(".");
    let re = RegexBuilder::new(&pattern).size_limit(1_000_000_000).dfa_size_limit(30).build().unwrap();
    let text = fs::read_to_string("../data/stripped_enwik9.txt").unwrap();
    let mut i = 0;
    let mut j = 10000;
    loop {
        if let Some(m) = re.find(&text[i..]) {
            i += m.end();
        }
        while i > j {
            println!("{j}");
            j += 10000;
        }
    }
}
*/
