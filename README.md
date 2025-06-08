# Tumor_Classifier_Model
This Is My First Trained Model Using Scikit-learn (LogisticRegression)


I Dont KNow How To Start So HERE WE GO AGAIN!

libraries:

```python
import sklearn as sk
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Modules From Sklearn(Scikit-learn):

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

I Used Dataset From [Kaggle](https://www.kaggle.com)

[Interesting Data to Visualize](https://www.kaggle.com/datasets/alexisbcook/data-for-datavis?select=cancer.csv)-Cancer.csv

Don't Know Why He Thinks Cancer Is "INTERESTING" Thing To Happen!

cancer_df.head(5): Cant fit All Columns

![Check The ScreenShot Folder, If Not Loaded](ScreenShot/Cancer.head().png)


All The Features From Cancer.csv
```python
Id, Diagnosis, Radius (mean), Texture (mean), Perimeter (mean)

Area (mean), Smoothness (mean), Compactness (mean), Concavity (mean)

Concave points (mean), Symmetry (mean), Fractal dimension (mean)

Radius (se), Texture (se), Perimeter (se), Area (se)

Smoothness (se), Compactness (se), Concavity (se), Concave points (se)

Symmetry (se), Fractal dimension (se)

Radius (worst), Texture (worst), Perimeter (worst), Area (worst)

Smoothness (worst), Compactness (worst), Concavity (worst)

Concave points (worst), Symmetry (worst), Fractal dimension (worst)
```

