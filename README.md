# SIF Embedding

## Usage

```
from sif import SifEmbedding

sif = SifEmbedding("/path/to/model.bin")
sif.fit(["今日は天気が良い", "今日は天気が悪い", "明日は平日だ", "今日は気分が良い"])
a = sif.predict("明日は雨だ", 3)
```
