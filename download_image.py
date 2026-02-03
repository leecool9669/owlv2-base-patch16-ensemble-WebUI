# -*- coding: utf-8 -*-
"""下载 HF 页面缩略图到 template/images（使用代理）。"""
import os
import urllib.request

IMAGE_URL = "https://cdn-thumbnails.hf-mirror.com/social-thumbnails/models/google/owlv2-base-patch16-ensemble.png"
OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
OUT_PATH = os.path.join(OUT_DIR, "owlv2-base-patch16-ensemble.png")
PROXY = "http://127.0.0.1:18081"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    urllib.request.install_opener(opener)
    try:
        urllib.request.urlretrieve(IMAGE_URL, OUT_PATH)
        print("Downloaded:", OUT_PATH)
    except Exception as e:
        try:
            urllib.request.install_opener(urllib.request.build_opener())
            urllib.request.urlretrieve(IMAGE_URL, OUT_PATH)
            print("Downloaded (no proxy):", OUT_PATH)
        except Exception as e2:
            print("Failed:", e2)

if __name__ == "__main__":
    main()
