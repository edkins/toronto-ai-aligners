# Cleaning the xml and wikitext

This assumes you already have a file called `data/enwik9`.

```shell
cd enwikclean
cargo run --release
cd ..
```

This will create a file called `data/stripped_enwik9.txt`


# Creating vocab list

```shell
cd vocab
cargo run --release
cd ..
```

This will create a file called `data/vocab_enwik9.txt`


# Splitting the corpus into tokens

```shell
cd split
cargo run --release
cd ..
```

This will create a file called `tokens_enwik9`


# Training the transformer

```shell
python3 batch_transformer.py train
```

This will create a visualization called `data/params.svg`, and a params file called `data/model.pth`.
