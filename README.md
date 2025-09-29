# Molecule PL Wavelength Prediction

分子の発光波長（Photoluminescence wavelength, PL）を予測するための機械学習パイプラインです。  
Excel/CSV 形式の分子データから記述子を計算し、機械学習モデルで予測・解釈を行います。  

Qiita記事:  
- [化学×AI: 機械学習でPL波長を予測する（第1回：記述子計算とデータ可視化）](https://qiita.com/Osarunokagoya/items/cc0f79e3a3d959de8635)  
- [化学×AI: 機械学習でPL波長を予測する（第2回：予測モデルの構築）](https://qiita.com/Osarunokagoya/items/fbf618c29ba6c0f85253)  
- [化学×AI: 機械学習でPL波長を予測する（第3回：出力結果の妥当性判断）](https://qiita.com/Osarunokagoya/items/d822d0516b10f9ffa20e)  

---

## セットアップ方法

### 1. 簡単版（pip）
```bash
pip install -r requirements.txt
jupyter lab
```

### 2. 完全再現版（Docker/DevContainer）
- `.devcontainer/` ディレクトリに Dockerfile と environment.yml を同梱しています  
- VS Code Dev Containers を利用すると GPU 環境込みで再現可能です  

---

## データ
- 入力: `data/material_data.csv`  
- 記述子出力: `outputs/descriptors/*.csv`  
- 予測結果: `outputs/predictions/*.csv`  

※ `material_data.csv` はリポジトリに含まれているため、clone してすぐに実行可能です。  

---

## ノートブックの流れ
1. **1.preparation.ipynb**  
   - データ読み込み、SMILES 構造の前処理  
   - 記述子の計算（RDKit, Mordred, 3D）  

2. **2.EDA.ipynb**  
   - データの可視化・統計解析  

3. **3.Regression.ipynb / 3.Regression_xxx.ipynb**  
   - モデル構築（RF, XGBoost, LightGBM, NNなど）  

4. **5.predict_xxx.ipynb**  
   - テストデータ予測  
   - Applicability Domain (AD) を考慮した判定  

5. **6.SHAP.ipynb**  
   - モデル解釈（SHAP値による特徴量重要度の可視化）  

---

## 使用技術
- Python (3.9)  
- RDKit, Mordred  
- scikit-learn, LightGBM, XGBoost  
- Optuna（ハイパーパラメータ探索）  
- SHAP（解釈可能性）  
- PyTorch

---

## ライセンス
本リポジトリは研究・教育目的での利用を想定しています。
