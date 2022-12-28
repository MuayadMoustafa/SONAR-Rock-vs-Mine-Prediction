```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
```


```python
df = pd.read_csv('data/Machine_Learning_Projects-main/SONAR_Rock_vs_Mine_Prediction/Copy of sonar data - Copy of sonar data.csv', header=None)
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>R</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.0187</td>
      <td>0.0346</td>
      <td>0.0168</td>
      <td>0.0177</td>
      <td>0.0393</td>
      <td>0.1630</td>
      <td>0.2028</td>
      <td>0.1694</td>
      <td>0.2328</td>
      <td>0.2684</td>
      <td>...</td>
      <td>0.0116</td>
      <td>0.0098</td>
      <td>0.0199</td>
      <td>0.0033</td>
      <td>0.0101</td>
      <td>0.0065</td>
      <td>0.0115</td>
      <td>0.0193</td>
      <td>0.0157</td>
      <td>M</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.0323</td>
      <td>0.0101</td>
      <td>0.0298</td>
      <td>0.0564</td>
      <td>0.0760</td>
      <td>0.0958</td>
      <td>0.0990</td>
      <td>0.1018</td>
      <td>0.1030</td>
      <td>0.2154</td>
      <td>...</td>
      <td>0.0061</td>
      <td>0.0093</td>
      <td>0.0135</td>
      <td>0.0063</td>
      <td>0.0063</td>
      <td>0.0034</td>
      <td>0.0032</td>
      <td>0.0062</td>
      <td>0.0067</td>
      <td>M</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0.0522</td>
      <td>0.0437</td>
      <td>0.0180</td>
      <td>0.0292</td>
      <td>0.0351</td>
      <td>0.1171</td>
      <td>0.1257</td>
      <td>0.1178</td>
      <td>0.1258</td>
      <td>0.2529</td>
      <td>...</td>
      <td>0.0160</td>
      <td>0.0029</td>
      <td>0.0051</td>
      <td>0.0062</td>
      <td>0.0089</td>
      <td>0.0140</td>
      <td>0.0138</td>
      <td>0.0077</td>
      <td>0.0031</td>
      <td>M</td>
    </tr>
    <tr>
      <th>206</th>
      <td>0.0303</td>
      <td>0.0353</td>
      <td>0.0490</td>
      <td>0.0608</td>
      <td>0.0167</td>
      <td>0.1354</td>
      <td>0.1465</td>
      <td>0.1123</td>
      <td>0.1945</td>
      <td>0.2354</td>
      <td>...</td>
      <td>0.0086</td>
      <td>0.0046</td>
      <td>0.0126</td>
      <td>0.0036</td>
      <td>0.0035</td>
      <td>0.0034</td>
      <td>0.0079</td>
      <td>0.0036</td>
      <td>0.0048</td>
      <td>M</td>
    </tr>
    <tr>
      <th>207</th>
      <td>0.0260</td>
      <td>0.0363</td>
      <td>0.0136</td>
      <td>0.0272</td>
      <td>0.0214</td>
      <td>0.0338</td>
      <td>0.0655</td>
      <td>0.1400</td>
      <td>0.1843</td>
      <td>0.2354</td>
      <td>...</td>
      <td>0.0146</td>
      <td>0.0129</td>
      <td>0.0047</td>
      <td>0.0039</td>
      <td>0.0061</td>
      <td>0.0040</td>
      <td>0.0036</td>
      <td>0.0061</td>
      <td>0.0115</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 61 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>...</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029164</td>
      <td>0.038437</td>
      <td>0.043832</td>
      <td>0.053892</td>
      <td>0.075202</td>
      <td>0.104570</td>
      <td>0.121747</td>
      <td>0.134799</td>
      <td>0.178003</td>
      <td>0.208259</td>
      <td>...</td>
      <td>0.016069</td>
      <td>0.013420</td>
      <td>0.010709</td>
      <td>0.010941</td>
      <td>0.009290</td>
      <td>0.008222</td>
      <td>0.007820</td>
      <td>0.007949</td>
      <td>0.007941</td>
      <td>0.006507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022991</td>
      <td>0.032960</td>
      <td>0.038428</td>
      <td>0.046528</td>
      <td>0.055552</td>
      <td>0.059105</td>
      <td>0.061788</td>
      <td>0.085152</td>
      <td>0.118387</td>
      <td>0.134416</td>
      <td>...</td>
      <td>0.012008</td>
      <td>0.009634</td>
      <td>0.007060</td>
      <td>0.007301</td>
      <td>0.007088</td>
      <td>0.005736</td>
      <td>0.005785</td>
      <td>0.006470</td>
      <td>0.006181</td>
      <td>0.005031</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001500</td>
      <td>0.000600</td>
      <td>0.001500</td>
      <td>0.005800</td>
      <td>0.006700</td>
      <td>0.010200</td>
      <td>0.003300</td>
      <td>0.005500</td>
      <td>0.007500</td>
      <td>0.011300</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000800</td>
      <td>0.000500</td>
      <td>0.001000</td>
      <td>0.000600</td>
      <td>0.000400</td>
      <td>0.000300</td>
      <td>0.000300</td>
      <td>0.000100</td>
      <td>0.000600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.013350</td>
      <td>0.016450</td>
      <td>0.018950</td>
      <td>0.024375</td>
      <td>0.038050</td>
      <td>0.067025</td>
      <td>0.080900</td>
      <td>0.080425</td>
      <td>0.097025</td>
      <td>0.111275</td>
      <td>...</td>
      <td>0.008425</td>
      <td>0.007275</td>
      <td>0.005075</td>
      <td>0.005375</td>
      <td>0.004150</td>
      <td>0.004400</td>
      <td>0.003700</td>
      <td>0.003600</td>
      <td>0.003675</td>
      <td>0.003100</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.022800</td>
      <td>0.030800</td>
      <td>0.034300</td>
      <td>0.044050</td>
      <td>0.062500</td>
      <td>0.092150</td>
      <td>0.106950</td>
      <td>0.112100</td>
      <td>0.152250</td>
      <td>0.182400</td>
      <td>...</td>
      <td>0.013900</td>
      <td>0.011400</td>
      <td>0.009550</td>
      <td>0.009300</td>
      <td>0.007500</td>
      <td>0.006850</td>
      <td>0.005950</td>
      <td>0.005800</td>
      <td>0.006400</td>
      <td>0.005300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.035550</td>
      <td>0.047950</td>
      <td>0.057950</td>
      <td>0.064500</td>
      <td>0.100275</td>
      <td>0.134125</td>
      <td>0.154000</td>
      <td>0.169600</td>
      <td>0.233425</td>
      <td>0.268700</td>
      <td>...</td>
      <td>0.020825</td>
      <td>0.016725</td>
      <td>0.014900</td>
      <td>0.014500</td>
      <td>0.012100</td>
      <td>0.010575</td>
      <td>0.010425</td>
      <td>0.010350</td>
      <td>0.010325</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.137100</td>
      <td>0.233900</td>
      <td>0.305900</td>
      <td>0.426400</td>
      <td>0.401000</td>
      <td>0.382300</td>
      <td>0.372900</td>
      <td>0.459000</td>
      <td>0.682800</td>
      <td>0.710600</td>
      <td>...</td>
      <td>0.100400</td>
      <td>0.070900</td>
      <td>0.039000</td>
      <td>0.035200</td>
      <td>0.044700</td>
      <td>0.039400</td>
      <td>0.035500</td>
      <td>0.044000</td>
      <td>0.036400</td>
      <td>0.043900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 208 entries, 0 to 207
    Data columns (total 61 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   0       208 non-null    float64
     1   1       208 non-null    float64
     2   2       208 non-null    float64
     3   3       208 non-null    float64
     4   4       208 non-null    float64
     5   5       208 non-null    float64
     6   6       208 non-null    float64
     7   7       208 non-null    float64
     8   8       208 non-null    float64
     9   9       208 non-null    float64
     10  10      208 non-null    float64
     11  11      208 non-null    float64
     12  12      208 non-null    float64
     13  13      208 non-null    float64
     14  14      208 non-null    float64
     15  15      208 non-null    float64
     16  16      208 non-null    float64
     17  17      208 non-null    float64
     18  18      208 non-null    float64
     19  19      208 non-null    float64
     20  20      208 non-null    float64
     21  21      208 non-null    float64
     22  22      208 non-null    float64
     23  23      208 non-null    float64
     24  24      208 non-null    float64
     25  25      208 non-null    float64
     26  26      208 non-null    float64
     27  27      208 non-null    float64
     28  28      208 non-null    float64
     29  29      208 non-null    float64
     30  30      208 non-null    float64
     31  31      208 non-null    float64
     32  32      208 non-null    float64
     33  33      208 non-null    float64
     34  34      208 non-null    float64
     35  35      208 non-null    float64
     36  36      208 non-null    float64
     37  37      208 non-null    float64
     38  38      208 non-null    float64
     39  39      208 non-null    float64
     40  40      208 non-null    float64
     41  41      208 non-null    float64
     42  42      208 non-null    float64
     43  43      208 non-null    float64
     44  44      208 non-null    float64
     45  45      208 non-null    float64
     46  46      208 non-null    float64
     47  47      208 non-null    float64
     48  48      208 non-null    float64
     49  49      208 non-null    float64
     50  50      208 non-null    float64
     51  51      208 non-null    float64
     52  52      208 non-null    float64
     53  53      208 non-null    float64
     54  54      208 non-null    float64
     55  55      208 non-null    float64
     56  56      208 non-null    float64
     57  57      208 non-null    float64
     58  58      208 non-null    float64
     59  59      208 non-null    float64
     60  60      208 non-null    object 
    dtypes: float64(60), object(1)
    memory usage: 99.2+ KB
    


```python
df[60].value_counts()
```




    M    111
    R     97
    Name: 60, dtype: int64




```python
sns.countplot(x=60, data=df)
```




    <AxesSubplot:xlabel='60', ylabel='count'>




    
![png](output_5_1.png)
    



```python
df.groupby(60).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
    <tr>
      <th>60</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>0.034989</td>
      <td>0.045544</td>
      <td>0.050720</td>
      <td>0.064768</td>
      <td>0.086715</td>
      <td>0.111864</td>
      <td>0.128359</td>
      <td>0.149832</td>
      <td>0.213492</td>
      <td>0.251022</td>
      <td>...</td>
      <td>0.019352</td>
      <td>0.016014</td>
      <td>0.011643</td>
      <td>0.012185</td>
      <td>0.009923</td>
      <td>0.008914</td>
      <td>0.007825</td>
      <td>0.009060</td>
      <td>0.008695</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>R</th>
      <td>0.022498</td>
      <td>0.030303</td>
      <td>0.035951</td>
      <td>0.041447</td>
      <td>0.062028</td>
      <td>0.096224</td>
      <td>0.114180</td>
      <td>0.117596</td>
      <td>0.137392</td>
      <td>0.159325</td>
      <td>...</td>
      <td>0.012311</td>
      <td>0.010453</td>
      <td>0.009640</td>
      <td>0.009518</td>
      <td>0.008567</td>
      <td>0.007430</td>
      <td>0.007814</td>
      <td>0.006677</td>
      <td>0.007078</td>
      <td>0.006024</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 60 columns</p>
</div>




```python
X = df.drop(columns= 60 , axis = 1 ) 
y = df[60]
```


```python
X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2 , shuffle = True , random_state = 1 , stratify= y)
```


```python
train_data_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(train_data_prediction , y_train)
```


```python
str(train_data_accuracy * 100) + ' %'
```




    '84.33734939759037 %'




```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```


```python
tree_clf = DecisionTreeClassifier(random_state=100)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    ================================================
    Accuracy Score: 100.00%
    _______________________________________________
    CLASSIFICATION REPORT:
                  M     R  accuracy  macro avg  weighted avg
    precision   1.0   1.0       1.0        1.0           1.0
    recall      1.0   1.0       1.0        1.0           1.0
    f1-score    1.0   1.0       1.0        1.0           1.0
    support    89.0  77.0       1.0      166.0         166.0
    _______________________________________________
    Confusion Matrix: 
     [[89  0]
     [ 0 77]]
    
    Test Result:
    ================================================
    Accuracy Score: 69.05%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy  macro avg  weighted avg
    precision   0.764706   0.640000  0.690476   0.702353      0.705322
    recall      0.590909   0.800000  0.690476   0.695455      0.690476
    f1-score    0.666667   0.711111  0.690476   0.688889      0.687831
    support    22.000000  20.000000  0.690476  42.000000     42.000000
    _______________________________________________
    Confusion Matrix: 
     [[13  9]
     [ 4 16]]
    
    


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 4332 candidates, totalling 12996 fits
    Best paramters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'random'})
    Train Result:
    ================================================
    Accuracy Score: 86.75%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy   macro avg  weighted avg
    precision   0.838384   0.910448   0.86747    0.874416      0.871811
    recall      0.932584   0.792208   0.86747    0.862396      0.867470
    f1-score    0.882979   0.847222   0.86747    0.865100      0.866393
    support    89.000000  77.000000   0.86747  166.000000    166.000000
    _______________________________________________
    Confusion Matrix: 
     [[83  6]
     [16 61]]
    
    Test Result:
    ================================================
    Accuracy Score: 73.81%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy  macro avg  weighted avg
    precision   0.761905   0.714286  0.738095   0.738095      0.739229
    recall      0.727273   0.750000  0.738095   0.738636      0.738095
    f1-score    0.744186   0.731707  0.738095   0.737947      0.738244
    support    22.000000  20.000000  0.738095  42.000000     42.000000
    _______________________________________________
    Confusion Matrix: 
     [[16  6]
     [ 5 15]]
    
    


```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    ================================================
    Accuracy Score: 100.00%
    _______________________________________________
    CLASSIFICATION REPORT:
                  M     R  accuracy  macro avg  weighted avg
    precision   1.0   1.0       1.0        1.0           1.0
    recall      1.0   1.0       1.0        1.0           1.0
    f1-score    1.0   1.0       1.0        1.0           1.0
    support    89.0  77.0       1.0      166.0         166.0
    _______________________________________________
    Confusion Matrix: 
     [[89  0]
     [ 0 77]]
    
    Test Result:
    ================================================
    Accuracy Score: 78.57%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy  macro avg  weighted avg
    precision   0.760000   0.823529  0.785714   0.791765      0.790252
    recall      0.863636   0.700000  0.785714   0.781818      0.785714
    f1-score    0.808511   0.756757  0.785714   0.782634      0.783866
    support    22.000000  20.000000  0.785714  42.000000     42.000000
    _______________________________________________
    Confusion Matrix: 
     [[19  3]
     [ 6 14]]
    
    


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = RandomizedSearchCV(estimator=rf_clf, scoring='f1',param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)

rf_cv.fit(X_train, y_train)
rf_best_params = rf_cv.best_params_
print(f"Best paramters: {rf_best_params})")

rf_clf = RandomForestClassifier(**rf_best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    D:\Programs\A\lib\site-packages\sklearn\model_selection\_search.py:969: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan]
      warnings.warn(
    

    Best paramters: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': True})
    Train Result:
    ================================================
    Accuracy Score: 100.00%
    _______________________________________________
    CLASSIFICATION REPORT:
                  M     R  accuracy  macro avg  weighted avg
    precision   1.0   1.0       1.0        1.0           1.0
    recall      1.0   1.0       1.0        1.0           1.0
    f1-score    1.0   1.0       1.0        1.0           1.0
    support    89.0  77.0       1.0      166.0         166.0
    _______________________________________________
    Confusion Matrix: 
     [[89  0]
     [ 0 77]]
    
    Test Result:
    ================================================
    Accuracy Score: 78.57%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy  macro avg  weighted avg
    precision   0.782609   0.789474  0.785714   0.786041      0.785878
    recall      0.818182   0.750000  0.785714   0.784091      0.785714
    f1-score    0.800000   0.769231  0.785714   0.784615      0.785348
    support    22.000000  20.000000  0.785714  42.000000     42.000000
    _______________________________________________
    Confusion Matrix: 
     [[18  4]
     [ 5 15]]
    
    


```python
n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 10]
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = GridSearchCV(rf_clf, params_grid, scoring="f1", cv=3, verbose=2, n_jobs=-1)


rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 768 candidates, totalling 2304 fits
    

    D:\Programs\A\lib\site-packages\sklearn\model_selection\_search.py:969: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan]
      warnings.warn(
    

    Best parameters: {'bootstrap': True, 'max_depth': 2, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    Train Result:
    ================================================
    Accuracy Score: 89.16%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy   macro avg  weighted avg
    precision   0.858586   0.940299  0.891566    0.899442      0.896489
    recall      0.955056   0.818182  0.891566    0.886619      0.891566
    f1-score    0.904255   0.875000  0.891566    0.889628      0.890685
    support    89.000000  77.000000  0.891566  166.000000    166.000000
    _______________________________________________
    Confusion Matrix: 
     [[85  4]
     [14 63]]
    
    Test Result:
    ================================================
    Accuracy Score: 69.05%
    _______________________________________________
    CLASSIFICATION REPORT:
                       M          R  accuracy  macro avg  weighted avg
    precision   0.695652   0.684211  0.690476   0.689931      0.690204
    recall      0.727273   0.650000  0.690476   0.688636      0.690476
    f1-score    0.711111   0.666667  0.690476   0.688889      0.689947
    support    22.000000  20.000000  0.690476  42.000000     42.000000
    _______________________________________________
    Confusion Matrix: 
     [[16  6]
     [ 7 13]]
    
    


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
```


```python
from sklearn.tree import DecisionTreeClassifier
tr = DecisionTreeClassifier(random_state= 42)
tr.fit(X_train, y_train) 
```




    DecisionTreeClassifier(random_state=42)




```python
tr.score(X_train, y_train)*100 
```




    100.0




```python
y_pred = tr.predict(X_test)
accuracy_score(y_pred, y_test)*100
```




    73.80952380952381




```python
tr = SVC(C = 1.0, kernel = 'linear')
tr.fit(X_train, y_train)
```




    SVC(kernel='linear')




```python
y_pred = tr.predict(X_test)
accuracy_score(y_pred, y_test)*100
```




    80.95238095238095




```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```




    LogisticRegression()




```python
y_pred = lr.predict(X_test)
accuracy_score(y_pred, y_test)*100
```




    78.57142857142857




```python
ac = neighbors.KNeighborsClassifier(n_neighbors=10)
ac.fit(X_train, y_train) 
```




    KNeighborsClassifier(n_neighbors=10)




```python
y_pred = ac.predict(X_test)
accuracy_score(y_pred, y_test)*100
```

    D:\Programs\A\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    




    69.04761904761905




```python
rf = RandomForestClassifier(n_estimators=2)
rf.fit(X_train, y_train)
```




    RandomForestClassifier(n_estimators=2)




```python
y_pred = rf.predict(X_test)
accuracy_score(y_pred, y_test)*100
```




    80.95238095238095




```python
input_data =np.asarray((0.0164,0.0173,0.0347,0.007,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.089,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.062,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.009,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.004)).reshape(1,-1)
prediction = lr.predict(input_data)

if(prediction == 'R') :
  print ("Peediction is Rock")
else : 
  print('prediction is Mine')
```

    prediction is Mine
    
