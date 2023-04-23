Download Link: https://assignmentchef.com/product/solved-10418-hw5-variational-inference
<br>
<h1>1          Written Questions</h1>

Answer the following questions in the template provided. Then upload your solutions to Gradescope. You may use L<sup>A</sup>T<sub>E</sub>X or print the template and hand-write your answers then scan it in. Failure to use the template may result in a penalty. There are 44 points and 19 questions.

<h2>1.1        Mean-Field Approximation for Multivariate Gaussians</h2>

In this question, we’ll explore how accurate a Mean-Field approximation can be for an underlying multivariate Gaussian distribution.

Assume we have observed data <strong>X</strong> that was drawn from a 2-dimensional Gaussian distribution <em>p</em>(<strong>x</strong>;<em>µ</em><em>, </em><strong>Λ</strong><sup>−1</sup>).

<em>p</em>(<strong>x</strong>;<em>µ</em><em>, </em>                                            (1.1)

Note here that we’re using the <em>precision </em>matrix <strong>Λ </strong>= <strong><sup>Σ</sup></strong><sup>−1</sup>. An additional property of the precision matrix is that it is symmetric, so Λ<sub>12 </sub>= Λ<sub>21</sub>. This will make your lives easier for the math to come.

We will approximate this 2-dimensional Gaussian with a mean field approximation, <em>q</em>(<strong>x</strong>) = <em>q</em>(<strong>x</strong><sub>1</sub>)<em>q</em>(<strong>x</strong><sub>2</sub>), the product of two 1-dimensional distributions <em>q</em>(<strong>x</strong><sub>1</sub>) and <em>q</em>(<strong>x</strong><sub>2</sub>). For now, we won’t assume any form for this distributions.

<ol>

 <li>(1 point) Short Answer: Write down the equation for log<em>p</em>(<strong>X</strong>). For now, you can leave all of the parameters in terms of vectors and matrices, not their subcomponents.</li>

 <li>(2 points) Short Answer: Group together everything that involves <strong>X</strong><sub>1 </sub>and remove anything involving <strong>X</strong><sub>2</sub>. We claim that there exists some distribution <em>q</em><sup>∗</sup>(<strong>X</strong>) = <em>q</em><sup>∗</sup>(<strong>X</strong><sub>1</sub>)<em>q</em><sup>∗</sup>(<strong>X</strong><sub>2</sub>) that minimizes the KL divergence <em>q</em><sup>∗ </sup>= argmin<em><sub>q </sub></em>KL(<em>q</em>||<em>p</em>). And further, said distribution will have a component <em>q<sup>?</sup></em>(<strong>X</strong><sub>1</sub>) will be proportional to the quantity you find below.</li>

</ol>

It can be shown that this implies that <em>q</em>(<strong>X</strong><sub>1</sub>) (and therefore <em>q</em>(<strong>X</strong><sub>2</sub>)) is a Gaussian distribution.

Where

Using these facts, we’d like to explore how well our approximation can model the underlying distribution.

<ol start="3">

 <li>Suppose the parameters of the true distribution are <em>µ</em> and .

  <ul>

   <li>(1 point) Numerical Answer: What is the value of the mean of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>1</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the variance of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>1</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the mean of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>2</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the variance of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>2</sub>)?</li>

   <li>(2 points) Plot: Provide a <em>computer-generated </em>contour plot to show the result of our approximation <em>q</em><sup>∗</sup>(<strong>X</strong>) and the true underlying Gaussian <em>p</em>(<strong>X</strong>;<em>µ</em><em>,</em><strong>Λ</strong>) for the parameters given above.</li>

  </ul></li>

 <li>Suppose the parameters of the true distribution are <em>µ</em> and .

  <ul>

   <li>(1 point) Numerical Answer: What is the value of the mean of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>1</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the variance of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>1</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the mean of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>2</sub>)?</li>

   <li>(1 point) Numerical Answer: What is the value of the variance of the Gaussian for <em>q</em><sup>∗</sup>(<strong>X</strong><sub>2</sub>)?</li>

   <li>(2 points) Plot: Provide a <em>computer-generated </em>contour plot to show the result of our approximation <em>q</em><sup>∗</sup>(<strong>X</strong>) and the true underlying Gaussian <em>p</em>(<strong>X</strong>;<em>µ</em><em>,</em><strong>Λ</strong>) for the parameters given above.</li>

  </ul></li>

 <li>(1 point) Describe in words how the plots you generated provide insight into the behavior of minimization of <em>KL</em>(<em>q</em>||<em>p</em>) with regards to the low probability and high probability regions of the the true vs. approximate distributions.</li>

</ol>

<h2>1.2        Variational Inference for Gaussian Mixture Models</h2>

Now that we have seen how the mean-field approximation works for a multivariate Gaussian, let’s look at the case of Gaussian Mixture Models. Suppose we have a Bayesian mixture of unit-variance univariate Gaussian distributions. This mixture consists of 2 components each corresponding to a Gaussian distribution, with means <em>µ </em>= {<em>µ</em><sub>1</sub><em>,µ</em><sub>2</sub>}. The mean parameters are drawn independently from a Gaussian prior distribution N(0<em>,σ</em><sup>2</sup>). The prior variance <em>σ</em><sup>2 </sup>is a hyperparameter. Generating an observation <em>x<sub>i </sub></em>from this model is done according to the following generative story:

<ol>

 <li>Choose a cluster assignment <em>c<sub>i </sub></em>for the observation. The cluster assignment is chosen from the distribution Categorical and indicates which latent cluster <em>x<sub>i </sub></em>comes from. Encode <em>c<sub>i </sub></em>as a one-hot vector where [1<em>,</em>0] indicates that <em>x<sub>i </sub></em>is assigned to cluster 0 and vice versa.</li>

 <li>Generate <em>x<sub>i </sub></em>from the corresponding Gaussian distribution N(<em>c<sup>T</sup><sub>i </sub></em><em>µ</em><em>,</em>1)</li>

</ol>

The complete hierarchical model is as follows:

<em>µ<sub>k </sub></em>∼ N(0<em>,σ</em><sup>2</sup>)<em>,k </em>∈ {1<em>,</em>2}

<em>c<sub>i </sub></em>∼ Categorical <em>x<sub>i</sub></em>|<em>c<sub>i</sub>,</em><em>µ </em>∼ N(<em>c<sup>T</sup><sub>i </sub></em><em>µ</em><em>,</em>1)<em>,i </em>∈ [1<em>,n</em>]

where n is the number of observations generated from the model.

<ol>

 <li>(1 point) What are the observed and latent variables for this model?</li>

 <li>(1 point) Write down the joint probability of observed and latent variables under this model</li>

 <li>(3 points) Let’s calculate the ELBO (evidence lower-bound) for this model. Recall that the ELBO is given by the following equation:</li>

</ol>

ELBO(<em>q</em>) = E<em>q</em>[log<em>p</em>(<strong>x</strong><em>,</em><strong>z</strong>)] − E<em>q</em>[log<em>q</em>(<strong>z</strong>)]

To calculate <em>q</em>(<strong>z</strong>), we will now use the mean-field assumption. Under this assumption, each latent variable is governed by its own latent factor, resulting in the following probability distribution:

!

Here <em>q</em>(<em>µ<sub>k</sub></em>;<em>m<sub>k</sub>,v<sub>k</sub></em><sup>2</sup>) is the Gaussian distribution for the <em>k</em>-th mixture component with mean and variance <em>m<sub>k </sub></em>and <em>v<sub>k</sub></em><sup>2</sup>. <em>q</em>(<em>c<sub>i</sub></em>;<em>a<sub>i</sub></em>) is the categorical distribution for the <em>i</em>-th observation with assignment probabilities <em>a<sub>i </sub></em>(<em>a<sub>i </sub></em>is a 2-dimensional vector). Given this assumption, write down the ELBO as a function of the variational parameters <strong>m</strong><em>,</em>.

<ol start="4">

 <li>Now that we have the ELBO formulation, let’s try to compute coordinate updates for our latent variables. Remember that the optimal variational density of a latent variable <em>z<sub>i </sub></em>is proportional to the exponentiated expected log of the complete conditional given all other latent variables in the model and the observed data. In other words:</li>

</ol>

!

<em>q<sub>i</sub></em>(<em>z<sub>i</sub></em>) ∝ exp E−<em><sub>j</sub></em>[log<em>p</em>(<em>z<sub>j</sub></em>|<strong>z</strong><sub>−<em>j</em></sub><em>,</em><strong>x</strong>)]

Equivalently, you can also say that the variational density is proportional to the exponentiated expected log of the joint E−<em><sub>j</sub></em>[log<em>p</em>(<em>z<sub>j</sub>,</em><strong>z</strong><sub>−<em>j</em></sub><em>,</em><strong>x</strong>)]. This is a valid coordinate update since the expectations on the right side of the equation do not involve <em>z<sub>j </sub></em>due to the mean-field assumption.

<ul>

 <li>(4 points) Show that the variational update for .</li>

</ul>

(Hint: We can write the optimal variational density for cluster assignment variables as

!

<em>q</em>(<em>c<sub>i</sub></em>;<em>a<sub>i</sub></em><sub>1</sub>) ∝ exp               log<em>p</em>(<em>c<sub>i</sub></em>) + E<em>µ</em>[log<em>p</em>(<em>x<sub>i</sub></em>|<em>c<sub>i</sub>,</em><em>µ</em>);<strong>m</strong><em>,</em><strong>v</strong><sup>2</sup>] .           Feel free to drop added constants

<ul>

 <li>(6 points) Show that the variational updates for the <em>k</em>-th mixture component are and.</li>

</ul>

(Hint: We can write the optimal variational density for the <em>k</em>-th mixture component as

!

. Feel free to drop added constants

<h2>1.3        Running CAVI: Toy Example</h2>

Let’s now see this in action!

Recall that the CAVI update algorithm for a Gaussian Mixture Model is as follows:

Note that our notation differs slightly, with <em>ϕ </em>corresponding to <em>a </em>and <em>s</em><sup>2 </sup>corresponding to <em>v</em><sup>2</sup>. We also have <em>K </em>= 2. Assume initial parameters, <strong>m </strong>= [0<em>.</em>5<em>,</em>0<em>.</em>5], <strong>v</strong><sup>2 </sup>= [1<em>,</em>1] and <em>a<sub>i </sub></em>= [0<em>.</em>3<em>,</em>0<em>.</em>7] for all <em>i </em>∈ <em>n </em>and a sample <em>x </em>= [0<em>.</em>1<em>,</em>−0<em>.</em>3<em>,</em>1<em>.</em>2<em>,</em>0<em>.</em>8<em>,</em>−0<em>.</em>5]. Also assume prior variance <em>σ</em><sup>2 </sup>= 0<em>.</em>01

Write a python script implementing the above procedure and run it for 5 epochs. You should submit your code to autolab as a .tar file named cavi.tar containing a single file cavi.py. You can create that file by running:

tar -cvf cavi.tar cavi.py

from the directory containing your code.

After the fifth epoch, report

<ol>

 <li>(2 points) The variational parameters <strong>m</strong>.</li>

</ol>

<table width="215">

 <tbody>

  <tr>

   <td width="30"><strong>m</strong></td>

   <td width="92"> </td>

   <td width="92"> </td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>(2 points) The variational parameters <strong>v</strong><sup>2</sup>.</li>

</ol>

<table width="216">

 <tbody>

  <tr>

   <td width="32"><strong>v</strong>2</td>

   <td width="92"> </td>

   <td width="92"> </td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>(2 points) The variational parameters <strong>a</strong>.</li>

</ol>

Hint:

<ol>

 <li>Note that the expectation update for <strong>a </strong>does not depend on <em>µ</em>. (Why?)</li>

 <li>The expectation of the square of a Gaussian random variable is E[<em>X</em><sup>2</sup>] = <em>V ar</em>[<em>X</em>] + E([<em>X</em>])<sup>2</sup>.</li>

</ol>

<h2>1.4        Variational Inference vs. Monte Carlo Methods</h2>

Let’s end with a brief comparison between variational methods and MCMC methods. We have seen that both classes of methods can be used for learning in scenarios involving latent variables, but both have their own sets of advantages and disadvantages. For each of the following statements, specify whether they apply more suitably to VI or MCMC methods:

<ol>

 <li>(1 point) Transforms inference into optimization problems.</li>

</ol>

Variational Inference

MCMC

<ol start="2">

 <li>(1 point) Is easier to integrate with back-propagation.</li>

</ol>

Variational Inference

MCMC

<ol start="3">

 <li>(1 point) Involves more stochasticity.</li>

</ol>

Variational Inference

MCMC

<ol start="4">

 <li>(1 point) Non-parametric.</li>

</ol>

Variational Inference

MCMC

<ol start="5">

 <li>(1 point) Is higher variance under limited computational resources.</li>

</ol>

Variational Inference

MCMC

<h2>1.5        Wrap-up Questions</h2>

<ol>

 <li>(1 point) Multiple Choice: Did you correctly submit your code to Autolab?</li>

</ol>

Yes

No

<ol start="2">

 <li>(1 point) Numerical answer: How many hours did you spend on this assignment?.</li>

</ol>