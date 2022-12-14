use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while, take},
    combinator::{opt, recognize, verify},
    multi::many0,
    sequence::tuple,
};
use std::collections::HashMap;
use std::fs;

fn article_separator(i: &[u8]) -> IResult<&[u8], &[u8]> {
    tag(b"########")(i)
}

fn word(i: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(tuple((
        opt(tag(b" ")),
        verify(byte, |c:&[u8]|matches!(c[0],  b'a'..=b'z' | b'A'..=b'Z')),
        take_while(|c| matches!(c, b'a'..=b'z')),
    )))(i)
}

fn byte(i: &[u8]) -> IResult<&[u8], &[u8]> {
    take(1usize)(i)
}

fn item(i: &[u8]) -> IResult<&[u8], &[u8]> {
    alt((word, article_separator, byte))(i)
}

fn canonicalize(mut word: &[u8]) -> Vec<u8> {
    if word.len() <= 1 {
        return word.to_vec();
    }
    let mut result = vec![];
    if word[0] == b' ' {
        word = &word[1..];
    }
    for ch in word {
        if *ch >= b'A' && *ch <= b'Z' {
            result.push(*ch + 32);
        } else {
            result.push(*ch);
        }
    }
    result
}

fn main() {
    let bytes = fs::read("../data/stripped_enwik9.txt").unwrap();
    let mut freqs = HashMap::new();
    println!("Parsing words");
    let (_,words) = many0(item)(&bytes).unwrap();
    println!("Computing word frequencies. {} words", words.len());
    for (i,word) in words.iter().enumerate() {
        *freqs.entry(canonicalize(word)).or_insert(0u64) += 1;
        if i > 0 && i % 1000000 == 0 {
            println!("{i}");
        }
    }
    println!("Sorting by freq");
    let mut freq_list:Vec<_> = freqs.iter().collect();
    freq_list.sort_by_key(|(_k,v)|-(**v as i64));
    let mut vocab_string = vec![];
    println!("Making vocab string");
    let mut count = 0;
    for (k,_v) in freq_list {
        if k.len() > 1 {
            vocab_string.extend_from_slice(k);
            vocab_string.push(b'\n');
            count += 1;
            if count > 1000000 {
                break;
            }
        }
    }
    fs::write("../data/vocab_enwik9.txt", vocab_string).unwrap();
}

