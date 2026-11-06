Predicting Perceived Personality Traits from Face Photographs

Code for the coursework "Using Machine Learning Methods to Recognize Human Psychological Traits from a Photograph". The coursework report is written in Russian; this README summarizes it in English.

At a glance: 2,222 face photos · 40 binary perceived-trait labels · CLIP ViT-B/32 image embeddings (512-d) · logistic regression with PCA · mean per-trait accuracy 0.7737 on the 34 best-predicted traits (0.7562 on all 40).

Research question

Which machine-learning model predicts certain perceived psychological traits of a person most accurately from a photo of their face?

Scope and responsible use

The labels are perceived traits: ratings given by human observers who looked at each photo. They are not self-reported personality scores and say nothing verified about the person pictured. A model that scores well here has learned how a crowd of raters judges faces, including the effects of facial expression and stereotypes. This is a coursework exercise, not a validated personality assessment, and it should not be used to make decisions about real people.

Data
Source: the 10k US Adult Faces Database (Bainbridge, Isola & Oliva, 2013). It contains 10,168 face photos; 2,222 of them come with psychology attributes. This project uses those 2,222 photos.
Annotations: 40 attributes, formed from 20 traits and their dictionary antonyms. Each is rated from 1 (not expressed) to 9 (fully expressed). Every photo was rated by 30 people: one group of 15 rated 20 of the attributes and another group of 15 rated the other 20.
Traits (20 pairs): typical-atypical, interesting-boring, calm-aggressive, common-uncommon, intelligent-unintelligent, kind-mean, responsible-irresponsible, trustworthy-untrustworthy, attractive-unattractive, caring-cold, emotionally stable-unstable, emotional-unemotional, familiar-unfamiliar, friendly-unfriendly, happy-unhappy, egotistic-humble, memorable-forgettable, normal-weird, sociable-introverted, confident-uncertain. The choice builds on the 14 dimensions of face evaluation found by Oosterhof & Todorov (2008), plus typical, familiar and memorable.
Photos: adults whose gender, race and age distribution is close to that of the US population. The set excludes minors, people wearing glasses, heavy makeup or visible deformities. Images are cropped to the face oval, 72 ppi, 256 px high.
Not included here. The authors share the database on request for research use under a license agreement (see their page), so the photos, annotation tables and derived embeddings are not part of this repository.
Method
Step	What the notebook does
1. Labels	Loads the Excel annotation table with pandas, drops control questions and rater metadata, averages the ratings per photo and trait, then binarizes each trait at its sample-wide mean (1 = above average, 0 = below). Result: a 2,222 x 40 binary matrix.
2. Class balance	The mean gap between the two class sizes across traits is 266 photos (12% of 2,222), so classes are roughly balanced and plain accuracy is used.
3. Features	embetter pipeline: ImageLoader (RGB) then ClipEncoder, which resizes images to 224 x 224 and encodes them with CLIP ViT-B/32 into a 512-d vector. Result: a 2,222 x 512 feature matrix.
4. Split	80% train / 20% test (1,777 / 445 photos), random_state=42.
5. Model comparison	For each trait, nine scikit-learn / XGBoost classifiers with default settings are fitted independently: KNN, SVM, decision tree, random forest, naive Bayes, LDA, logistic regression, AdaBoost, XGBoost. Logistic regression is the model taken forward.
6. Multi-label setup	MultiOutputClassifier fits one independent classifier per trait. Exact-match accuracy over 40 labels is too strict, so the metric is the mean per-trait accuracy: accuracy computed for each trait separately, then averaged.
7. Tuning	GridSearchCV with 5-fold cross-validation over a pipeline of StandardScaler, PCA and logistic regression. The search runs in two stages because some penalty/solver pairs are incompatible: an L2 grid (PCA components 8-128, C 0.1-10, class weight, six solvers, tolerance) and an L1 grid (PCA components 8-64, C 0.01-1, liblinear/saga, tolerance).
8. Trait filtering	ROC AUC is computed per trait. The six traits below 0.75 are dropped and the search is repeated on the remaining 34.
Results
Model	Traits	Mean per-trait accuracy
Logistic regression, default hyperparameters (max_iter=1000)	40	0.746
After grid search: L2, C=0.1, newton-cholesky, tol=1e-3, 64 PCA components	40	0.7562
After removing the 6 weakest traits and repeating the search: L1, saga, C=0.1, tol=1e-5	34	0.7737

The last row covers a smaller set of traits, so it is not directly comparable with the first two rows. Figures are as reported in the coursework. The 64 PCA components keep 0.892 of the variance of the embeddings.

All 40 traits had ROC AUC above 0.5. The six dropped traits, mostly related to how memorable a face is rather than to personality impressions:

Trait	ROC AUC
familiar	0.7389
uncommon	0.7322
atypical	0.7298
unfamiliar	0.7159
common	0.6975
calm	0.6835
Limitations
Small dataset (2,222 photos). Neural networks are the usual tool for image prediction, and fine-tuning a network pre-trained on millions of images would probably give higher accuracy. With this little data, the project stays with classical models on top of pre-trained embeddings. Fine-tuning is the natural next step.
Faces show emotions. Some labels (for example emotional-unemotional) reflect the expression at the moment of the photo rather than stable traits.
US faces only. The photos come from US residents, so the model should not be assumed to work for other populations.
Low image resolution (72 ppi). This may lower model quality, although it also speeds up processing.
Repository contents
File	Description
code_work.ipynb	The full pipeline, in sections: Libraries, Dataset, Preprocessing, Choosing the model, Logistic regression, Results. Saved cell outputs are cleared.
README.md	This file.
Reproducing
Request the database from its authors and keep the face images and psychology-attributes.xlsx.
Install the dependencies:
bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl pillow "embetter[sbert]"

scikit-learn 1.2 or newer is needed for the newton-cholesky solver. embetter[sbert] also installs sentence-transformers, which ClipEncoder uses; the default model is clip-ViT-B-32. 3. The notebook was written for Google Colab with the data on Google Drive. For a local run, remove the google.colab cell and change the paths in the Dataset section (all_images_dir, needed_images_dir, the Excel path) and the output paths for dataframe_for_learning.csv, embeddings.npy and labels.csv. 4. Run the sections top to bottom. The embedding step is the slowest; its output is saved to embeddings.npy and labels.csv, which later cells reload. 5. The last cells predict traits for two example photos. Point them at your own image files.
