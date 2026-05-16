"LogisticRegression" 
1.why we need to move from linear to logistic:

Lin Reg predicts continuos values
Ex :
house price pred , salary , marks , etc..
but if we want :
1.pass/fail , 2. spam/not spam ,etc..
 * These are the classification prblm:
Lin Reg may predict:
1.8
-2.5
0.8
Which does not make sense for the classification
so we need :
o/p Between 0 and 2
that is why we use the LR
---------------------------------------
2.why it is called  logistic regression:


LR uses Lin Reg internally
z=mx+c
then it applies a special function called:
- this it applies a special function called:
  Sigmoid function:
   z= 1/(1+e^(-z))
this converts values from ;
-infi to +infi
 into : 0 to 1

Hence it performs classification using regression math

.........................................................
3.math behind LR:

                                                                              z= mx + c
lets:
m=3                                                       where;
c=-6                                                m=slope
                                                       x=input
then;                                               c=constant/intercept
z= 2x - 6
code:
x=4
z= (m* x)+c
print("m value:",m)
print("x value:",x)
print("z value:",z)
print("c value:",c)

simoid Function calcu:
import math 
z=2
sigmoid = 1/(1+math.exp(-z))
print("z values:",z)
print("sigmoid probability:",sigmoid)

If probability > 0.5:
class = 1
else:
class = 0 

---------------------------------------------
4.sigmoid function with calculation
--------------------------------------------

# visualizing the sigmoid curve:

 import matplotlib.pyplot as plt
 import numpy as np
  x= np.linespace(-10,10,100)
  y= 1/(1+np.exp(-x))

plt.figure(figsize=(9,5))
plt.plot(x,y)

plt.xlabel('z values")
plt.ylabel("sigmoid output")
plt.title("sigmoid curve")

plt.grid(True)

plt.show()


--------------------------------------------
5.understanding m and c values
-------------------------------------------
6.confusion matrix:
actual           predicted                result 
1                     1                        true pos 
1                     0                        false pos
0                     1                        false neg
0                     0                        true neg

------------------------------------------
7. evalution metrices:

1.accuracy=  tp + tn / total
2.precision= tp / tp + fp
3.recall = tp / tp + fn
4.f1-score = 2*precision * recall / precision + recall

------------------------------------------
8.end - to - end LR:

student pass/fail prediction

features:
study hours:
target:
pass or fail

