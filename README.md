# 鐵達尼號生存者預測_演算法預測
  - 1092_數學軟體應用期末報告作業
  - 基於sklearn進行預測
  
  ---
## 1. 準備 Preparing
### 1.1. 載入模組 Module Loading
 - seaborn 是基於 matlotlib 的數據圖像化套件
 - re 在這裡負責將字串做重新編一分類
 - sklearn 為本次使用的演算法模組庫 
   * accuracy_score 用於評估準確度
   * train_test_split 用於將訓練資料整理成 sklearn 的通用格式
   
---
### 1.2. 載入和查看資料 Data Loading and Checking
---
## 2. 資料數值化 Data Numeralization
### 2.1. 空欄位填滿 Null Table Filling
 - 訓練資料和測試資料上存在缺漏需要填補

#### 2.1.1. 查看資料缺漏 Checking Data Loss
 - 可以得知在Age年齡、Cabin船艙房號、Embarked登船港口上有缺漏
---
#### 2.1.2. 填補房號資料 Filling Carbin
 - 先查看Carbin房號是否和Pclass船票等級是否有相關性
    - 房號的號碼是1位英文字母+2位或3位數字
 - 我們先用未缺漏的資料查看相關性，確定它是可以使用的資料，再做填補
---
#### 2.1.3. 填補年齡資料 Filling Age
   - 有關一個鐵達尼號倖存者的知識，當年船上的乘客會將救生船優先讓給婦女及小孩，所以年齡資料可能會是重要依據。
   - 先查看未缺陷資料的狀況。
##### 將年齡分級來看相關性
 - 顯然直接使用數字年齡不合適，所以將年齡數字分級簡化數值再看一遍。
    - 用純粹每10歲分級、和用未成年和成年分級看看
  - 因此我們在這邊不選擇年齡資料
---
#### 2.1.4. 填補票價資料 Filling Fare
 - 票價資料有漏一個.........，直接劃記為0
 - 嘗試對票價做分級
---
### 2.2. 資料數值化 Data Numeralization 
#### 2.2.1. 性別 Sex、直系親屬(父母子女)人數 Parch、同輩親屬及配偶數量 SibSp
---
#### 2.2.2. 船票編號 Ticket
---
#### 2.2.3. 登船處 Embarked
---
#### 2.2.4. 姓名 Name 
 - 死亡flag分類
   - 一定死的：'Capt.', 'Don.', 'Jonkheer.', 'Rev.'
   - 倖存機率低的：'Mr.', 'Dr.'
   - 倖存機率高的：'Col.', 'Major.', 'Master.', 'Miss.', 'Mrs.'
   - 一定活的：'Countess.', 'Lady.', 'Mlle.', 'Mme.', 'Ms.', 'Sir.'
  - "填補房號資料 Filling Carbin"
        sns.heatmap(checkSetTrain1.corr(), annot = True)
        'Cabin', 'Pclass', 'Deck_C', 'Deck_B', 'Deck_D', 'Deck_E', 'Deck_A', 'Deck_F', 'Deck_G', 'Deck_T'
        
 - "填補年齡資料 Filling Age"
        sns.heatmap(checkSetTrain21.corr(), annot = True)
        'Age'
        sns.heatmap(checkSetTrain22.corr(), annot = True)
        'Age'
        sns.heatmap(checkSetTrain23.corr(), annot = True)
        'Age'
        
 - "填補票價資料 Filling Fare"
```
        sns.heatmap(checkSetTrain3.corr(), annot = True)
        'Fare'
```       
 - "性別 Sex、直系親屬(父母子女)人數 Parch、同輩親屬及配偶數量 SibSp"
```
        sns.heatmap(checkSetTrain4.corr(), annot = True)
        'Sex', 'Parch', 'SibSp'
```        
 - "船票編號 Ticket"
```
        sns.heatmap(checkSetTrain5.corr(), annot = True)
        'ticketType'
```     
 - "登船處 Embarked"
```
        sns.heatmap(checkSetTrain6[['Embarked', 'Survived']].corr(), annot = True)
        sns.heatmap(checkSetTrain6[['embarkedS', 'embarkedC', 'embarkedQ', 'Survived']].corr(), annot = True)
        'Embarked', 'embarkedS', 'embarkedC', 'embarkedQ'
```
 - "姓名 Name"
```
        sns.heatmap(checkSetTrain7[['Title', 'Survived']].corr(), annot = True)
        'Title'
```        

  - 我們準備一組帶有
    **[票價Fare, 性別Sex, 直系親屬人數Parch, 同輩親屬人數Sibsp, 票等ticketType, 是否自S港口上船embarkedS, 是否自C港口上船embarkedC, 是否自Q港口上船embarkedQ, 頭銜Title]**
<!--     作為第一組資料 -->
     - TrainSet1, TestSet1
     
<!--   - 再分類一組有 -->

### 資料格式處理

---
## 3. 演算法實作 Algorithm Implementation
### 3.1. 決策樹 Decision Tree
#### 不限制max_depth
#### 限制max_depth=10

### 3.2. 隨機森林 Random Forest

### 3.3. LogisticRegression

### 結論
        就我們目前所使用來自sklearn三個不同的演算法，他們都在同一筆數據下得到了相近的準確度，尤其決策樹在限制max_depth之後反而得到較高準確度，顯然是因為過擬合導致，用max_depth進行剪枝成功迴避了這個問題。
