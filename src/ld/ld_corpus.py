from pathlib import Path

topics = {"dokujo-tsushin": 0,
          "it-life-hack": 1,
          "kaden-channel": 2,
          "livedoor-homme": 3,
          "movie-enter": 4,
          "peachy": 5,
          "smax": 6,
          "sports-watch": 7,
          "topic-news": 8}


def extract_text(file):
    with open(file) as f:
        lines = f.readlines()[2:]
    return "".join([line.strip() for line in lines if line.strip()])


def load_ld_corpus():
    docs = []
    labels = []
    for topic, label in topics.items():
        for file in Path(f"data/text/{topic}").glob(f"{topic}-*.txt"):
            text = extract_text(file)
            # texts[topic].append(text)
            docs.append(text)
            labels.append(label)
    return docs, labels
