# google colabで実行
from icrawler.builtin import BingImageCrawler

# オバマ大統領の画像を検索 → obamaフォルダに格納
crawler = BingImageCrawler(storage={"root_dir": "obama"})
crawler.crawl(keyword="オバマ大統領", max_num=500)

# obamaフォルダをzip形式に変更 → ローカルにダウンロード
!zip -r /content/download.zip /content/obama
from google.colab import files
files.download("/content/download.zip")

# ノッチも同様
crawler = BingImageCrawler(storage={"root_dir": "notch"})
crawler.crawl(keyword="オバマ　ノッチ", max_num=500)
!zip -r /content/download_notch.zip /content/notch
files.download("/content/download_notch.zip")