#%%
import matplotlib.pyplot as plt
import tikzplotlib
#import matplotlib2tikz
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#%%

# generate random data-set
np.random.seed(0)
x = np.random.rand(50, 1)
y = 2 + 3 * x + np.random.rand(50, 1)

#%%
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x, y)
#%%
# Make predictions using the testing set
y_pred = regr.predict(x)
#%%
# The coefficients
w =  regr.coef_[0][0]
b = regr.intercept_[0]
print('w: ',w)
print('b: ', b )
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y, y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
#%%
# Plot outputs
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(111)
ax.scatter(x, y,  color='black')
ax.plot(x, y_pred, color='blue', linewidth=3)
plt.style.use("ggplot")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.set_title("Simple linear regression model  $y=w_1 x + b$")
ax.grid(True)
#ax.text(0.1, 0.9,'$w_1 =' +'{:.2f}'.format(w) +'$ \n $b ='+'{:.2f}'.format(b) +'$', ha='center', va='center',transform=ax.transAxes,  bbox=dict(facecolor='none', edgecolor='black'))
#plt.xticks(())
#plt.yticks(())
tikzplotlib.save("C:\\Users\\ga45tis\\GIT\\masterthesisgeneral\\latex\\900 Report\\images\\fundamentels\\simple_linear_regression.tex")
plt.show()