# House-Prices
## კონკურსის მიმოხილვა
ეს პროექტი შესრულებულია Kaggle-ის კონკურსის ფარგლებში — House Prices: Advanced Regression Techniques. კონკურსის მიზანია სახლების გაყიდვის ფასის პროგნოზირება 79 სხვადასხვა მახასიათებლის საფუძველზე, როგორებიცაა სახლის ფართობი, მდებარეობა, ხარისხი, მშენებლობის წელი და სხვა.
შეფასების მეტრიკაა RMSLE (Root Mean Squared Log Error) — რაც ნიშნავს, რომ პროგნოზი ლოგარითმულ სკალაზე ფასდება.
## ჩემი მიდგომა
პრობლემა გადავჭერი ეტაპობრივად: ჯერ მონაცემები გავასუფთავე და დავამუშავე, შემდეგ ჩავატარე feature selection, და ბოლოს რამდენიმე სხვადასხვა მოდელი გავტესტე. ყველა ექსპერიმენტი დალოგილია MLflow-ზე Dagshub-ის საშუალებით.
___
## რეპოზიტორიის სტრუქტურა
    
     House-Prices/
                ├── model_experiment.ipynb   # მთავარი notebook — cleaning, feature engineering, training
                ├── model_inference.ipynb    # საბოლოო პროგნოზი და submission
                └── README.md                # README
___
## Feature Engineering
NaN მნიშვნელობების დამუშავება
მონაცემებში საკმაოდ ბევრი გამოტოვებული მნიშვნელობა იყო. თითოეულ სვეტს ინდივიდუალურად მივადექი, რადგან NaN-ის მიზეზი ყველგან სხვადასხვა იყო:

* **PoolQC, Fence, MiscFeature, Alley** — ამ სვეტებში NaN-ების 90%+ იყო. ეს ნიშნავდა, რომ სახლს უბრალოდ არ გააჩნდა ეს ობიექტი. სვეტები მთლიანად წავშალე, რადგან მათი ინფორმაციური ღირებულება მინიმალური იყო.
* **LotFrontage** — შევავსე მედიანით, რადგან მედიანა გამოტოვებული მნიშვნელობების შევსებისას უკეთესია ვიდრე საშუალო — ნაკლებად ექვემდებარება outlier-ების გავლენას.
* **MasVnrType, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond** — ამ სვეტებში NaN ნიშნავდა, რომ სახლს ეს ელემენტი საერთოდ არ ჰქონია (მაგ. სარდაფი, ავტოფარეხი, ბუხარი). ამიტომ 'None' სტრინგით შევავსე.
* **MasVnrArea** — რიცხვითი სვეტი, სადაც NaN ნიშნავდა 0 კვ.ფ-ს. შევავსე 0-ით.
* **GarageYrBlt** — შევავსე მედიანით.
* **Electrical — მხოლოდ 1 NaN იყო. ვივარაუდე, რომ სახლს ელექტრობა ჰქონდა, მაგრამ მონაცემი არ ჩაიწერა, ამიტომ ყველაზე გავრცელებული მნიშვნელობით (mode) შევავსე.


### კატეგორიული ცვლადების კოდირება
**Ordinal Encoding** — ხარისხობრივი სვეტებისთვის, სადაც კატეგორიებს შორის ლოგიკური თანმიმდევრობა არსებობს, გამოვიყენე ხელით შედგენილი რუქა:
Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
ეს მიდგომა გამოვიყენე შემდეგ სვეტებზე: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond.
One-Hot Encoding — დანარჩენი კატეგორიული სვეტებისთვის გამოვიყენე pd.get_dummies(). drop_first=True გამოვიყენე multicollinearity-ის თავიდან ასაცილებლად. შემდეგ .align() მეთოდით train და test set-ების სვეტები გავათანასწორე.

## **Cleaning** მიდგომები
კორელაციის ფილტრის გამოყენებით ვიპოვე ძალიან მაღლა კორელირებული სვეტების წყვილები (threshold=0.8). თითოეული წყვილიდან ის სვეტი ავირჩიე წასაშლელად, რომელსაც target-თან (SalePrice) სუსტი კავშირი ჰქონდა. საბოლოოდ წავშალე: GarageArea, TotRmsAbvGrd, 1stFlrSF.

## Feature Selection
### SelectKBest
One-hot encoding-ის შემდეგ 200+ სვეტი გამომივიდა, რაც ძალიან ბევრი ხმაურიანი feature-ია. SelectKBest(f_regression, k=50) გამოვიყენე სტატისტიკურად ყველაზე მნიშვნელოვანი 50 სვეტის შესარჩევად. ეს მნიშვნელოვნად გააუმჯობესა Decision Tree-ის შედეგი — ნაკლები ხმაური ნიშნავდა ნაკლებ overfitting-ს.
### Correlation Filter
ზემოთ აღწერილი კორელაციის ფილტრი ასევე Feature Selection-ის ნაწილია — არარელევანტური, დუბლირებული სვეტების ამოშლა მოდელს ეხმარება.

## Training
ტესტირებული მოდელები
სამი მოდელი გავტესტე, ყველა KFold cross-validation-ით (5 fold):
| მოდელი | cv_rmse_mean | test_rmse | შენიშვნა |
| :--- | :---: | :---: | :--- |
| **LinearRegression** | -- | -- | Baseline |
| **DecisionTreeRegressor (max_depth=30)** | 42,924 | 41,208 | Overfitting |
| **DecisionTreeRegressor (max_depth=10)** | 41,588 | 43,051 | საუკეთესო |

### **Hyperparameter** ოპტიმიზაცია
Decision Tree-ისთვის ორი განსხვავებული max_depth გავტესტე:

* max_depth=30 — ძალიან ღრმა ხე, რომელიც training მონაცემებს "ზეპირდება". test_rmse კარგი გამოვიდა, მაგრამ cv_rmse_mean უფრო მაღალი იყო — ეს overfitting-ის ნიშანია.
* max_depth=10 — შეზღუდული სიღრმის ხე. cv_rmse_mean test_rmse-ზე დაბალი გამოვიდა (41,588 vs 43,051), რაც ნიშნავს, რომ მოდელი სტაბილურია და კარგად განაზოგადებს ახალ მონაცემებზე.

საბოლოო მოდელის შერჩევა
საბოლოო მოდელად ავირჩიე DecisionTreeRegressor(max_depth=10) შემდეგი მიზეზებით:

1. cv_rmse_mean და test_rmse შორის სხვაობა მინიმალურია (~1,500) — მოდელი არ overfits
2. cross-validation score კარგია, რაც ნიშნავს სტაბილურ პერფორმანსს სხვადასხვა data split-ზე
3. max_depth=30 მოდელს უკეთესი test_rmse ჰქონდა, მაგრამ ეს სავარაუდოდ კონკრეტული random split-ის "იღბალია" და არა ჭეშმარიტი გაუმჯობესება

## არტეფაქტები და რეპროდუცირებადობა
იმისათვის, რომ `model_inference.ipynb` წარმატებით გაეშვას, საჭიროა წინასწარ დატრენინგებული მოდელები და სკალერები. ეს ფაილები ატვირთულია Kaggle Dataset-ის სახით:
[House Prices Model](https://www.kaggle.com/datasets/lukashekiladze/house-prices-model)

**ფაილები მოიცავს:**
* `model.pkl` — დატრენინგებული მოდელი.
* `scaler.pkl` — StandardScaler ობიექტი.
* `selector.pkl` — SelectKBest ობიექტი.
* `fill_values.pkl` — ლექსიკონი, რომელიც ინახავს LotFrontage-ის მედიანას და სხვა მნიშვნელობებს.

## MLflow Tracking
ექსპერიმენტების ბმული
ყველა ექსპერიმენტი დალოგილია Dagshub-ზე:
https://dagshub.com/lshek22/House-Prices
ჩაწერილი მეტრიკები და პარამეტრები
Cleaning ეტაპზე დავალოგე:

წაშლილი სვეტები და მიზეზი
თითოეული სვეტის შევსების სტრატეგია და კონკრეტული მნიშვნელობა (მედიანა, mode, 'None')

Feature Selection ეტაპზე დავალოგე:

კორელაციის threshold და წაშლილი სვეტები
encoding მეთოდი და საბოლოო სვეტების რაოდენობა

Training ეტაპზე თითოეული run-ისთვის დავალოგე:

* model_name — მოდელის სახელი
* max_depth — Decision Tree-ის სიღრმე
* features_count — გამოყენებული feature-ების რაოდენობა
* cv_rmse_mean — cross-validation-ის საშუალო შეცდომა
* cv_rmse_std — cross-validation შედეგების სტანდარტული გადახრა
* test_rmse — test set-ზე შეცდომა

საუკეთესო მოდელის შედეგები

* cv_rmse_mean: 41,588
* test_rmse: 43,051
* Kaggle Public Score: 0.20387
