# How-to-Talk-Linear-Regression-Optimizing-Loss-Function-Mean-Squared-Error

December 27, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- Last time, we got a little taste of linear regression,
by looking at the case of one-dimensional data.
Today, we'll move on to
full-fledged least-squares regression.
This is a very simple and powerful method,
and one of the cornerstones of statistics.
So we'll begin by formulating what it means
to do regression with multiple predictor variables.
We will phrase the learning problem
as an optimization task, with an explicit loss function.
And then we'll see that this optimization task,
is actually quite easy to solve.
So, as our running example, we will use data
from a diabetes study.
So the goal of this study was to asses what features
might impact the progression of the disease,
how much worse it gets
over the course of a fixed period of time, for example.
So data was collected from 442 diabetes patients.
For each patient, 10 features were measured,
features that could plausibly impact
the progression of the disease.
So they were age, sex, body mass index,
which is a measure of how overweight somebody is,
average blood pressure,
and various blood serum measurements.
So these 10 features were measured for each of the patients,
and then, the patient came back a year later,
and a value was measured which indicated
how much the disease had progressed
over the course of the year.
And that was, that value was called y.
This is a classic regression problem.
We have a response value y,
and we want to be able to predict this response
in terms of a bunch of predictor variables,
and we have 10 of these, x1 through x10,
which we can package together into a 10-dimensional vector.
So we'd like to model y, as a linear function of the x's.
So let's see what this means.
So when we have just one variable, x, let's say,
a linear function of the xs,
a linear function of x would be something like this.
Y equals wx plus b, where w is the slope,
and b is the intercept.
When we have 10 variables,
a linear function looks like this.
We have w1x1 plus w2x2, all the way to w10x10,
and then again, we have the intercept term at the end.
Now we can write this a little bit more simply,
as w dot x plus b, where what we've done
is to package together the coefficients w1 through w10,
into a single vector w.
So this is a nice compact form of the linear function.
So that's the sort of model
that we're gonna fit to the data.
Now, on any given point x, our prediction is gonna be
w dot x plus b, and the correct value is y.
So the error, we have to figure out some way to penalize it,
and we're gonna use squared loss,
so the squared difference between the value we predict,
and the actual value.
And this leads immediately
to the least-squares regression problem.
So we have end data points,
in our case there are 442 patients, so n equals 442.
The points x1 through x10 are d-dimensional vectors,
for us there are 10 features, so d equals 10,
and then we also have the response values y1 through yn.
Now based on this data, we wanna learn a linear function.
In other words, we want to learn parameters w and b.
And we wanna pick the linear function
that minimizes the total squared error,
so that minimizes the sum over all the data points,
of the squared error on the ith data point.
So a very simple loss function.
We wanna find the vector w, and the value b
that minimize this loss function.
Now it will turn out that there is a nice closed form
for the optimal w and b, and we'll see that very soon.
But first, let's just see what happens on the diabetes data.
So suppose we had no predictor variables at all,
we just had to predict y,
what value of y would we predict?
Well we'd predict the average value of y, the mean value.
And in this case, the mean squared error
would simply be the variance in y.
So in the diabetes example,
this variance turns out to be 5930.
Now this might seem a little large,
but actually it turns out these y values,
the progression of the disease,
are numbers like 100, 200, 300.
So when y is 200, y squared is 40,000.
So relative to that, a variance of about 6,000
seems plausible.
Now let's say we throw in one predictor feature,
let's say we use body mass index for example.
So what I've shown over here is a scatter plot
with body mass index on the x axis, and the y value,
the progression of the disease on the y axis.
So, this is a scatter plot of the 442 data points,
and I've also shown the line that is obtained
by least-squares regression.
By using this line,
the mean squared error drops significantly, down to 3,890.
Then we can throw in a second predictor variable,
let's say we use one of the serum measurements.
The mean squared error drops further.
And then we can go ahead and use all 10 variables,
to make the mean squared error the smallest of all, 2860.
Now computationally, all of these different
calculations are very simple,
because there's a nice closed formula for w.
And in fact, this is one of the
few cases in this entire course
where there's such a simple solution
for the optimization tasks that we are dealing with.
So, so let's see how it works out.
So once again, we are fitting a linear function
to the data, the linear function is something of this form,
if we have d features.
Now, this term over here,
the intercept term, is important,
but it sticks out a little bit,
it's a little bit inconvenient
because it looks a little different from everything else.
And when we're doing the optimization, for example,
we have to treat it separately as a result.
Luckily it turns out
that there's a way to make it disappear,
basically by assimilating it into w.
So let's see how we do this.
So here's the trick for assimilating b into the w vector.
The first thing we do, is to add an extra feature
to the data x,
by just sticking a one in front of all the data points.
So if, for example, we have a data point
that looks like this,
we write down the same thing, but with a one in front of it.
So each point x now becomes a new point,
which we'll call x twiddle, which is just one
followed by whatever x was before.
So these new points, x twiddle,
are now d plus one dimensional.
The next thing we do, is to stick b and w together,
and we call that vector w twiddle.
So now let's see what happens to our linear function,
so our linear function was w dot x plus b,
but this is exactly the same as w twiddle dot x twiddle.
Why, because w twiddle is become a w,
and x twiddle is one comma x so it's exactly b plus w dot x.
So we've managed to rewrite the linear function
without b, by adding an extra feature to the data points,
and also to the w vector.
So now, we need only optimize for w twiddle,
and here's the new loss function.
We wanna find the w twiddle that leads, once again,
to the least-squared error on the data set.
And as we'll see, there's a nice formula for w twiddle.
So the first step in deriving the formula
will actually be to re-write this loss function,
purely as a matrix vector product.
Why would we do this?
Well it turns out that it leads to a nice,
simple expression for w twiddle.
And so let's see what happens.
So, this is our loss function.
We're gonna rewrite it in terms of matrices and vectors,
and actually this is,
this is something that is often very useful to do.
Part of the reason for this is that
standard platforms like Python
have highly optimized routines for matrix math,
for matrix and vector multiplies.
At the same time, they can be extremely slow
on things like four loops.
So if you can somehow take your four loop,
and rewrite it as a matrix vector product,
or as a matrix matrix product,
your code could potentially speed up dramatically.
But how do we rewrite things in this way, well,
let's take a look at this loss function,
so we have a loss function that contains a summation,
now normally if we have a summation,
that would make us think of a four loop.
But we can rewrite it, and let's see how.
So, first, we create a matrix
with one data point per row, so we have n rows,
and, of course we have augmented our data,
now we've added this extra feature,
we've stuck ones in the front of each vector.
So, we have d plus one columns.
So that's the matrix x.
And let's create a vector with all the response values,
so y is n by one.
Now, let's see what is x times w twiddle.
Well, it's the first row of x
dot product, it would w twiddle,
the second row of x dot product,
it would w twiddle and so on,
it's exactly the predicted values.
So this is exactly w twiddle dot x1,
w twiddle dot x2, and so on,
all the way to w twiddle dot xn.
And likewise, if we look at y minus xw twiddle,
what does that work out to?
It's simply the vector of errors.
It's the error on the first point,
y1 minus w twiddle dot x1,
all the way to the error on the last point,
yn minus w twiddle dot xn.
So in particular, the squared norm of this vector,
which is this quantity over here,
is exactly the sum of all the squared errors,
it's exactly our loss function.
So we've rewritten our loss function,
without any summation at all.
Purely in terms of vectors and matrices.
At this point, we can take the derivative,
and set it to zero, and it turns out
that the solution is the following.
We said w twiddle to x transpose x inverse,
that's a d plus one by d plus one matrix,
times x transpose y, which is a d plus one times y vector.
And, the result is d plus one times one.
And of course the very first term in that answer
is the intercept term that we were previously calling b.
So it's a very simple solution,
and that concludes our treatment
of least-squares regression.
It's a very easy method,
and it's a very powerful method as well,
that's useful in a wide range of applications.
This is the bread and butter of much of statistics.
Okay, see you next time.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-an-Introduction-to-Linear-Regression

https://github.com/noey2020/Hpw-to-Talk-More-Generative-Models

https://github.com/noey2020/How-to-Talk-Gaussian-Generative-Models

https://github.com/noey2020/How-to-Talk-Multivariate-Gaussian

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-3

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-2

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-1

https://github.com/noey2020/How-to-Talk-2D-Generative-Modeling

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
