# 勝ちパターン探知機(スマホ版)

過去に利益を出した銘柄(三菱電機・オムロン・キーエンス・ABEJA・円谷フィールズ など)に似た「底打ち→上昇」の形をしている日本株を、**毎日自動で**探してスマホに表示するアプリ。PCは不要。

## しくみ

```
【平日16:30 自動】GitHubの無料サーバー
   全銘柄(約3,900)の株価を取得 → 5年位置・3ヶ月底打ちを判定
   → 候補だけ 類似度・業績・ニュース(・AI評価)を追加
   → 結果を results.json にしてスマホ用ページに反映
        ↓
【スマホ】ホーム画面のアイコンから開くだけ
   テーマをタップ → 絞り込みの様子・候補・脱落銘柄・チャートが即表示
   (全銘柄ぶんの結果を持っているので、テーマを変えても待ち時間なし)
```

- 費用: 0円(GitHubの公開リポジトリは Actions も Pages も無料)
- AI評価だけは任意。Anthropic APIキーを登録した場合のみ従量課金。登録しなくても、画面の「Claudeに聞く」ボタンで普段のClaudeに質問文ごと渡せる

## 最初の設定(1回だけ・PC推奨、スマホでも可)

### 1. GitHubアカウント

<https://github.com/signup> で作成(メールアドレスだけでOK)。作ったか覚えてない場合は <https://github.com/login> で普段のメールアドレスを入れて「Forgot password?」を試すと分かる。

### 2. リポジトリを作る

右上の「＋」→「New repository」
- Repository name: `stock-app`(好きな名前でOK)
- **Public** を選ぶ(無料でページ公開するのに必要)
- 「Create repository」

### 3. ファイルをアップロード

1. 受け取った `stock-app-mobile` フォルダ(zipなら解凍したもの)を開く
2. 作ったリポジトリの画面で「uploading an existing file」をクリック
3. フォルダの**中身を全部**(`screener` `web` `config` `data` `setup` フォルダと各ファイル)ドラッグ&ドロップ →「Commit changes」
4. 自動実行の設定ファイルを作る(ここだけ手打ち):
   「Add file」→「Create new file」→ 名前欄に `.github/workflows/screen.yml` と入力
   → `setup/screen.yml` の中身を全部コピーして貼り付け →「Commit changes」

> `.github` は特殊なフォルダで、ドラッグでは上がらないことが多いので、4.の方法で作るのが確実。
> ファイル一覧に `.github` フォルダが出ていればOK。

### 4. ページ公開をオンにする

「Settings」→ 左メニュー「Pages」→ Source を **GitHub Actions** に変更

### 5. 1回目の計算を動かす

「Actions」タブ →(確認が出たら緑のボタンで有効化)→ 左の「screen」→ 右の「Run workflow」→「Run workflow」

30分〜1時間ほどで緑のチェックになれば完了。

### 6. スマホに入れる

`https://<GitHubのユーザー名>.github.io/stock-app/` をスマホで開く
- iPhone(Safari): 共有ボタン →「ホーム画面に追加」
- Android(Chrome): 右上︙ →「ホーム画面に追加」/「アプリをインストール」

以降は平日16:30に自動で更新される。すぐ更新したい時は、アプリ下の「↻ 今すぐ再計算」から Run workflow。

### 7.(任意)AI評価をオンにする

「Settings」→「Secrets and variables」→「Actions」→「New repository secret」
- Name: `ANTHROPIC_API_KEY` / Secret: Anthropic のAPIキー

次回の計算から、候補の上位20件にAIスコアとコメントが付く(従量課金)。

## 注意

- リポジトリは公開(Public)になる。中身はプログラム・勝ちパターン銘柄の一覧(config/win_patterns.yaml)・スクリーニング結果だけで、個人情報は含まない。ページのURLを知っていれば誰でも見られる状態
- 株価は Yahoo Finance(yfinance)、銘柄一覧は JPX、ニュースは Googleニュース。無料データなので欠けたり遅れたりすることがある
- 60日間リポジトリに変更がないとGitHubが自動実行を止める仕様があるので、45日ごとに自動で空コミットして防いでいる
- 投資助言ではなく、自分で判断するための参考情報

## カスタマイズ(GitHubの画面上で直接編集できる。スマホでも可)

| やりたいこと | 編集するファイル |
|---|---|
| テーマを増やす・銘柄を足す | `screener/themes.py`(`THEME_NAME_KEYWORDS` と `THEME_GENRES`) |
| 判定の厳しさを変える(底値圏30%、下落15%、反発10% など) | `screener/signals.py` の上の方のパラメータ |
| 勝ちパターン銘柄・その底の日付 | `config/win_patterns.yaml`(`pattern_start` に底の日付を入れると類似度の精度が上がる) |
| 自動実行の時刻 | `.github/workflows/screen.yml` の `cron`(UTC表記。`30 7` = 日本時間16:30) |

編集して「Commit changes」すれば、次回の計算から反映される。

## ファイル構成

```
.github/workflows/screen.yml  毎日の自動実行の設定
screener/                     計算プログラム(Python)
  run.py        全体の流れ・results.json の書き出し
  data.py       銘柄一覧(JPX)と株価(yfinance)の取得
  signals.py    5年位置・3ヶ月底打ち・類似度の判定
  enrich.py     業績・ニュース・AI評価(候補のみ)
  themes.py     テーマとジャンル分け
config/win_patterns.yaml      勝ちパターン銘柄
data/tickers.csv              JPXから取れなかった時の予備の銘柄一覧
setup/screen.yml              .github/workflows/screen.yml に貼る用の控え
web/                          スマホ画面(index.html 1枚 + アイコン等)
```

画面だけ試したい時は、公開後のURLの末尾に `?demo` を付けると架空データのデモが見られる。
