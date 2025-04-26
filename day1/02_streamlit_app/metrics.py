import streamlit as st
import nltk
from janome.tokenizer import Tokenizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# 追加モジュール
import language_tool_python
from collections import Counter

# language-tool-python の初期化
try:
    # 日本語対応の文法チェッカーを初期化
    # デフォルトでは英語なので、日本語の場合は言語を指定
    language_tool = language_tool_python.LanguageTool('ja-JP')
    print("LanguageTool loaded successfully.") # デバッグ用
except Exception as e:
    st.warning(f"LanguageToolの初期化中にエラーが発生しました: {e}\n文法チェックは無効になります。")
    # ダミー関数を作成
    class DummyLanguageTool:
        def check(self, text):
            return []
    language_tool = DummyLanguageTool()

def calculate_metrics(answer, correct_answer):
    """回答と正解から評価指標を計算する"""
    word_count = 0
    bleu_score = 0.0
    similarity_score = 0.0
    relevance_score = 0.0
    grammar_score = 0.0  # 追加: 文法性スコア
    diversity_score = 0.0  # 追加: 多様性スコア

    if not answer: # 回答がない場合は計算しない
        return bleu_score, similarity_score, word_count, relevance_score, grammar_score, diversity_score

    # 単語数のカウント
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(answer))  # ← list() でイテレータをリストに変換
    word_count = len(tokens)
    
    # 追加: 文法性スコアの計算
    try:
        errors = language_tool.check(answer)
        # エラーが少ないほど高スコア（1.0が満点）
        if len(answer) > 0:
            # エラー文字数の割合を計算し、それを1から引く
            error_chars = sum(len(err.context) for err in errors)
            grammar_score = max(0, 1.0 - (error_chars / len(answer)))
        else:
            grammar_score = 0.0
    except Exception as e:
        # st.warning(f"文法スコア計算エラー: {e}")
        grammar_score = 0.0  # エラー時は0
    
    # 追加: 多様性スコアの計算 (Type-Token Ratio)
    try:
        # トークンから表層形を抽出
        token_surfaces = [token.surface for token in tokens]
        if token_surfaces:
            # 重複しない単語の数 / 全単語数
            unique_tokens = set(token_surfaces)
            diversity_score = len(unique_tokens) / len(token_surfaces)
            
            # より高度なバージョン: エントロピーベースの多様性
            # token_counts = Counter(token_surfaces)
            # total = len(token_surfaces)
            # entropy = -sum((count/total) * math.log2(count/total) for count in token_counts.values())
            # max_entropy = math.log2(total) if total > 0 else 0
            # diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity_score = 0.0
    except Exception as e:
        # st.warning(f"多様性スコア計算エラー: {e}")
        diversity_score = 0.0  # エラー時は0

    # 正解がある場合のみBLEUと類似度を計算
    if correct_answer:
        # 既存のコード...
        answer_lower = answer.lower()
        correct_answer_lower = correct_answer.lower()

        # BLEU スコアの計算
        try:
            reference = [nltk_word_tokenize(correct_answer_lower)]
            candidate = nltk_word_tokenize(answer_lower)
            # ゼロ除算エラーを防ぐ
            if candidate:
                bleu_score = nltk_sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) # 4-gram BLEU
            else:
                bleu_score = 0.0
        except Exception as e:
            # st.warning(f"BLEUスコア計算エラー: {e}")
            bleu_score = 0.0 # エラー時は0

        # コサイン類似度の計算
        try:
            vectorizer = TfidfVectorizer()
            # fit_transformはリストを期待するため、リストで渡す
            if answer_lower.strip() and correct_answer_lower.strip(): # 空文字列でないことを確認
                tfidf_matrix = vectorizer.fit_transform([answer_lower, correct_answer_lower])
                similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            else:
                similarity_score = 0.0
        except Exception as e:
            # st.warning(f"類似度スコア計算エラー: {e}")
            similarity_score = 0.0 # エラー時は0

        # 関連性スコア（キーワードの一致率などで簡易的に計算）
        try:
            answer_words = set(re.findall(r'\w+', answer_lower))
            correct_words = set(re.findall(r'\w+', correct_answer_lower))
            if len(correct_words) > 0:
                common_words = answer_words.intersection(correct_words)
                relevance_score = len(common_words) / len(correct_words)
            else:
                relevance_score = 0.0
        except Exception as e:
            # st.warning(f"関連性スコア計算エラー: {e}")
            relevance_score = 0.0 # エラー時は0

    return bleu_score, similarity_score, word_count, relevance_score, grammar_score, diversity_score

def get_metrics_descriptions():
    """評価指標の説明を返す"""
    return {
        "正確性スコア (is_correct)": "回答の正確さを3段階で評価: 1.0 (正確), 0.5 (部分的に正確), 0.0 (不正確)",
        "応答時間 (response_time)": "質問を投げてから回答を得るまでの時間（秒）。モデルの効率性を表す",
        "BLEU スコア (bleu_score)": "機械翻訳評価指標で、正解と回答のn-gramの一致度を測定 (0〜1の値、高いほど類似)",
        "類似度スコア (similarity_score)": "TF-IDFベクトルのコサイン類似度による、正解と回答の意味的な類似性 (0〜1の値)",
        "単語数 (word_count)": "回答に含まれる単語の数。情報量や詳細さの指標",
        "関連性スコア (relevance_score)": "正解と回答の共通単語の割合。トピックの関連性を表す (0〜1の値)",
        "効率性スコア (efficiency_score)": "正確性を応答時間で割った値。高速で正確な回答ほど高スコア",
        "文法性スコア (grammar_score)": "文法的な正確さの指標。文法エラーが少ないほど高スコア (0〜1の値)",
        "多様性スコア (diversity_score)": "語彙の多様性を表すType-Token Ratio (TTR)。重複せず使われている単語の割合 (0〜1の値)"
    }
