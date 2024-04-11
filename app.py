import streamlit as st
import MeCab
import re
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# MeCabの初期化
mecab_tagger = MeCab.Tagger()

def mecab_tokenizer(text):
    node = mecab_tagger.parseToNode(text)
    tokens = []
    while node:
        word = node.surface
        hinshi = node.feature.split(",")[0]
        if hinshi == "名詞":
            if (not word.isnumeric()) and (not re.match(r'^[\u3040-\u309F]+$', word)):
                tokens.append(word)
        node = node.next
    return tokens

def generate_wordcloud(text):
    tokens = mecab_tokenizer(text)
    word_freq = Counter(tokens)
    
    # フォントパスの設定（環境に合わせて変更してください）
    font_path = 'ipaexg.ttf'

    wc = WordCloud(background_color="white", font_path=font_path, width=800, height=400).generate_from_frequencies(word_freq)
    return wc.to_image()

def visualize_word_frequencies(text):
    tokens = mecab_tokenizer(text)
    word_freq = Counter(tokens)
    
    # 頻度の高い上位50語を取得
    common_words = word_freq.most_common(50)
    
    # 可視化
    words, frequencies = zip(*common_words)
    plt.figure(figsize=(10, 8))
    plt.bar(words, frequencies)
    plt.xlabel('単語')
    plt.ylabel('出現頻度')
    plt.xticks(rotation=45)
    plt.title('単語の出現頻度')
    st.pyplot(plt)


def create_downloadable_csv(data):
    """
    単語の出現頻度データからCSV形式の文字列を生成する。
    """
    # データフレームを作成
    df = pd.DataFrame(data, columns=['単語', '出現頻度'])
    # 頻出順にソート
    df.sort_values(by='出現頻度', ascending=False, inplace=True)
    # CSV文字列に変換
    return df.to_csv(index=False)



def make_wordcloud_and_visualize_frequencies():
    st.title("ワードクラウド生成アプリ＆単語頻度可視化")

    user_input = st.text_area("テキストを入力してください", "ここにテキストを入力")

    if st.button("ワードクラウドを生成"):
        wordcloud_image = generate_wordcloud(user_input)
        st.image(wordcloud_image, use_column_width=True)
        visualize_word_frequencies(user_input)
        
        # 単語の出現頻度リストを取得
        tokens = mecab_tokenizer(user_input)
        word_freq = Counter(tokens).most_common()
        
        # CSV形式の文字列に変換
        csv_string = create_downloadable_csv(word_freq)
        
        # ダウンロードボタンを表示
        st.download_button(
            label="出現頻度データをダウンロード",
            data=csv_string,
            file_name='word_frequencies.csv',
            mime='text/csv',
        )

# Streamlitアプリの実行
make_wordcloud_and_visualize_frequencies()
