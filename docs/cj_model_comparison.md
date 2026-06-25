\documentclass[11pt]{article}

\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{natbib}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}

\bibliographystyle{plainnat}

\newcommand{\given}{\,\vert\,}
\newcommand{\E}{\mathbb{E}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\blambda}{\boldsymbol{\lambda}}
\newcommand{\Prob}{\mathrm{P}}
\newcommand{\logit}{\operatorname{logit}}

\title{\bf Competing models for comparative judgement:\\ a mathematical summary and a practical framework for model comparison}
\author{}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
\noindent Comparative judgement (CJ) is an increasingly popular method for placing a set of objects, typically pieces of student work, on a quality scale by aggregating many pairwise decisions. Almost all reported CJ studies are analysed under a single model, usually Bradley--Terry, and the reliability of the resulting scale is summarised by a single statistic, the scale separation reliability. This practice obscures a question that is rarely asked directly: would the substantive conclusions of a CJ study change if a different but equally defensible model had been fitted, and if so, which statistical tools should be used to decide between the competing models? We summarise the principal models for comparative judgement within a single probabilistic framework, making explicit how they relate to one another and where they genuinely diverge. We then describe the main families of model comparison technique, namely likelihood-based information criteria, Bayesian model comparison, out-of-sample predictive scoring, reliability-based measures, and rank-correlation robustness analyses, and we set out how each can be implemented in practice. Broadly speaking, conclusions are robust to the choice of link function but can be sensitive to structural assumptions concerning ties, intransitivity, rater heterogeneity, and the adaptive selection of pairs. We argue that a defensible CJ analysis should fit several competing models to the same data and report agreement at the level of the decisions that matter, rather than relying on a single model and a single reliability coefficient.
\end{abstract}

\section{Introduction}

Comparative judgement provides an alternative to absolute marking for assessing performances whose quality is difficult to capture in a mark scheme, such as extended writing, mathematical problem solving, or portfolio work \citep{pollitt2012,thurstone1927}. Rather than scoring each object in isolation, assessors are shown pairs of objects and asked only to decide which of the two is better with respect to a stated construct. The aggregate of many such decisions is then used to estimate a quality parameter for each object, and hence a rank order and an interval scale. The method is motivated by the claim that humans make more reliable relative than absolute judgements, and by the observation that a pairwise decision imposes a lower cognitive load than the assignment of an absolute score \citep{pollitt2012}.

The statistical engine underneath CJ is a paired comparison model, and the choice of model is rarely scrutinised. In the educational literature the model is almost always Bradley--Terry, although it is frequently derived from the work of Thurstone and Rasch rather than from \citet{bradleyterry1952} directly, and the distinction matters for how reliability is reported \citep{bramley2015,verhavert2018ssr}. The dominant adaptive software reports a single reliability statistic, the scale separation reliability (SSR), and a CJ study is typically judged adequate if this statistic is high. This convention has attracted sustained criticism. \citet{bramley2015} demonstrated by simulation that the adaptive selection of pairs can inflate the SSR even when the underlying responses are essentially random, and subsequent work has shown that the statistic can overestimate reliability when raters operate on different underlying constructs or when the true variance of the quality parameters is small \citep{bramleyvitello2018,crompvoets2022,verhavert2018ssr}.

These criticisms concern the reliability statistic rather than the model, but they point to a deeper and largely unaddressed issue. If the apparent quality of a CJ scale depends on the design and on the analysis, then so too might the substantive conclusions, namely the rank order, the interval positions, and any grade boundaries or selection decisions derived from them. The central question we address is therefore twofold: do the conclusions of a comparative judgement study change under competing models, and which statistical approaches can be used to identify the best performing model for a given data set. We do not propose a new model. Instead we collect the relevant models within a common framework, we describe the model comparison machinery, and we set out a practical workflow for applying it.

The remainder of this document is organised as follows. Section~\ref{sec:models} summarises the models mathematically, beginning with the Thurstone and Bradley--Terry models and proceeding to the extensions that handle ties, rankings, rater heterogeneity, and intransitivity. Section~\ref{sec:estimation} describes estimation, since the choice between maximum likelihood, penalised, and Bayesian estimation interacts with the choice of model. Section~\ref{sec:comparison} sets out the model comparison techniques. Section~\ref{sec:practice} describes how these techniques are implemented in practice and proposes a workflow. Section~\ref{sec:discussion} concludes.

\section{Models for comparative judgement}
\label{sec:models}

Throughout we consider a set of $N$ objects labelled $1,\dots,N$, and we assign to each object $i$ a real-valued quality parameter $\lambda_i$. The data consist of a collection of pairwise comparisons. For a comparison of objects $i$ and $j$ we write $y=1$ if $i$ is judged better than $j$ and $y=0$ otherwise. The models below differ chiefly in the function that maps the difference in quality parameters to the probability of a given outcome, and in the additional structure they admit beyond a simple win or loss.

\subsection{The Thurstone model}

The earliest model is due to \citet{thurstone1927}, who framed comparative judgement as a problem of discriminating between noisy internal representations. Specifically, we assume that the perceived quality of object $i$ on any given occasion is a Gaussian random variable
\begin{equation}
X_i \sim \mathrm{N}(\lambda_i,\sigma^2),
\label{eq:thurstone-latent}
\end{equation}
where $\lambda_i$ denotes the mean quality of object $i$ and $\sigma^2$ is a common discriminal variance, with the representations of distinct objects taken to be independent. This is Thurstone's Case~V. Object $i$ is judged better than object $j$ whenever its sampled representation is the larger, so that the choice probability is
\begin{equation}
\Prob(i \succ j) = \Prob(X_i > X_j) = \Phi\!\left(\frac{\lambda_i-\lambda_j}{\sigma\sqrt{2}}\right),
\label{eq:thurstone-prob}
\end{equation}
where $\Phi$ is the standard normal distribution function and $i \succ j$ denotes the event that $i$ is preferred to $j$. The scale of $\sigma$ is not identifiable from comparison data alone, and it is conventional to set $\sigma\sqrt{2}=1$, which absorbs the discriminal variance into the quality parameters and reduces \eqref{eq:thurstone-prob} to a probit model for the difference $\lambda_i-\lambda_j$. The pairwise version with Gaussian noise is often called the Thurstone--Mosteller model, the second name acknowledging Mosteller's contribution to its statistical treatment \citep{mosteller1951}.

\subsection{The Bradley--Terry model}

The Bradley--Terry model replaces the normal distribution function in \eqref{eq:thurstone-prob} with the logistic function \citep{bradleyterry1952}. We associate with each object a positive strength $\pi_i = e^{\lambda_i}$ and assume that the probability that $i$ beats $j$ is proportional to the strength of $i$. This is given by
\begin{equation}
\Prob(i \succ j) = \frac{\pi_i}{\pi_i+\pi_j} = \frac{1}{1+e^{-(\lambda_i-\lambda_j)}} = \sigma(\lambda_i-\lambda_j),
\label{eq:bt-prob}
\end{equation}
where $\sigma(\cdot)$ denotes the standard logistic function and $\lambda_i=\log\pi_i$ is the quality parameter of object $i$ on the log scale. Equivalently, the model is a logistic regression on the difference in quality parameters, since $\logit \Prob(i\succ j) = \lambda_i-\lambda_j$. The logistic and probit links are numerically very close after a rescaling of the parameters, and in practice the Bradley--Terry and Thurstone--Mosteller models yield near-identical scale estimates \citep{agresti2002,varinfirth2024}. Bradley--Terry is generally preferred on grounds of tractability, since it admits straightforward maximum likelihood estimation together with likelihood-ratio tests and analytic standard errors.

For estimation it is convenient to write the likelihood explicitly. Let $n_{ij}$ be the number of times objects $i$ and $j$ are compared and $w_{ij}$ the number of those comparisons won by $i$. Assuming the comparisons are conditionally independent given the quality parameters, the likelihood is
\begin{equation}
L(\blambda) = \prod_{i<j} \left(\frac{\pi_i}{\pi_i+\pi_j}\right)^{w_{ij}} \left(\frac{\pi_j}{\pi_i+\pi_j}\right)^{n_{ij}-w_{ij}},
\label{eq:bt-lik}
\end{equation}
where the product is over all compared pairs and the binomial coefficients have been omitted as they do not depend on $\blambda$. The quality parameters are identifiable only up to an additive constant, and it is conventional to impose either $\sum_i \lambda_i = 0$ or $\lambda_1=0$. A finite and unique maximum likelihood estimate exists provided the directed graph of wins and losses is strongly connected, that is, provided every object beats and is beaten by some other object either directly or through a chain of intermediaries \citep{turnerfirth2012}.

\subsection{The Bradley--Terry--Luce model and the Rasch framing}

The Bradley--Terry model is the pairwise special case of a more general model of choice. \citet{luce1959} showed that the rule under which an item is selected from a set with probability proportional to its strength is the unique rule satisfying a natural independence axiom. We then have, for a choice set $S$,
\begin{equation}
\Prob(\text{$i$ chosen from } S) = \frac{\pi_i}{\sum_{k\in S}\pi_k},
\label{eq:luce}
\end{equation}
where the sum runs over the items in $S$. Setting $|S|=2$ recovers \eqref{eq:bt-prob}, which is why the model is often called Bradley--Terry--Luce. In the educational CJ literature the same mathematics is reached from a different direction, through the dichotomous Rasch model. The Rasch model writes the probability of a correct response as $\sigma(\theta-\delta)$ for a person ability $\theta$ and an item difficulty $\delta$, and a CJ comparison is treated as an item whose outcome is governed by $\sigma(\lambda_i-\lambda_j)$. The two derivations produce the same likelihood, but the Rasch framing treats the exercise as measurement rather than as the description of preference data, and it is this framing that motivates the Rasch-style separation reliability discussed in Section~\ref{sec:comparison} \citep{bramley2015,verhavert2018ssr}.

\subsection{Ties: the Davidson model}

The basic models force every comparison to yield a winner. In practice assessors frequently encounter pairs they find genuinely hard to separate, and forcing a decision in such cases discards information and increases fatigue. \citet{davidson1970} extended Bradley--Terry to accommodate ties by introducing a single nonnegative parameter. We then have
\begin{align}
\Prob(i \succ j) &= \frac{\pi_i}{\pi_i+\pi_j+\nu\sqrt{\pi_i\pi_j}}, \label{eq:davidson-win}\\[4pt]
\Prob(i \sim j) &= \frac{\nu\sqrt{\pi_i\pi_j}}{\pi_i+\pi_j+\nu\sqrt{\pi_i\pi_j}}, \label{eq:davidson-tie}
\end{align}
where $i \sim j$ denotes a tie and $\nu \geq 0$ governs the overall propensity for ties to occur. Setting $\nu=0$ recovers the Bradley--Terry model. Allowing ties reduces the cognitive burden on assessors, but it increases the complexity of the likelihood and, in the Bayesian treatments now available, the cost of inference \citep{forcedmarriage2022,bpcs2021}.

\subsection{Rankings: the Plackett--Luce model}

Some CJ designs ask assessors to rank order more than two objects at once, rather than to make a single binary choice. The natural model for such data is the Plackett--Luce model \citep{plackett1975,luce1959}, which treats a ranking as a sequence of choices, each made from the objects not yet ranked. For a ranking $\rho=(\rho_1,\dots,\rho_K)$ of $K$ objects, in which $\rho_1$ is placed first, the probability is
\begin{equation}
\Prob(\rho) = \prod_{k=1}^{K-1} \frac{\pi_{\rho_k}}{\sum_{m=k}^{K}\pi_{\rho_m}},
\label{eq:plackett-luce}
\end{equation}
where the $k$th factor is the probability that object $\rho_k$ is chosen as the best of those remaining in positions $k$ to $K$. For $K=2$ the Plackett--Luce model reduces exactly to Bradley--Terry \citep{turner2020plackettluce}. It is worth noting that the pairwise outcomes implied by a single ranking are not independent, since they derive from one ordering, and treating them as independent Bradley--Terry observations overstates the information content of the data.

\subsection{Rater heterogeneity}

The models above assume that every comparison is an equally reliable draw from a common process. This assumption is implausible when comparisons are collected from many assessors of varying expertise and attention. One remedy is a mixture model in which each rater $r$ is informed with some probability and guesses otherwise. Specifically, we assume that
\begin{equation}
\Prob(\text{$r$ judges } i \succ j) = q_r\,\sigma(\lambda_i-\lambda_j) + (1-q_r)\tfrac{1}{2},
\label{eq:rater-mixture}
\end{equation}
where $q_r \in [0,1]$ is a rater-specific reliability, so that with probability $q_r$ the rater follows the Bradley--Terry model and with probability $1-q_r$ chooses at random. A second and complementary approach introduces rater effects directly into the linear predictor, for example a severity or a discrimination parameter, in the manner of a facets model. Either device allows unreliable or idiosyncratic assessors to be downweighted rather than allowed to distort the common scale, and the rater parameters are themselves informative for quality assurance \citep{dittrich1998}.

\subsection{Intransitivity}

The Bradley--Terry and Thurstone models impose stochastic transitivity: a single latent scale orders the objects, so that if $i$ tends to beat $j$ and $j$ tends to beat $k$, then $i$ tends to beat $k$. Real judgement data sometimes violate this, either because the construct is genuinely multidimensional or because preferences depend on the particular pairing. A flexible way to capture such departures is to add a skew-symmetric interaction to the linear predictor, giving
\begin{equation}
\logit \Prob(i \succ j) = (\lambda_i-\lambda_j) + \gamma_{ij}, \qquad \gamma_{ij}=-\gamma_{ji},
\label{eq:intransitive}
\end{equation}
where $\gamma_{ij}$ represents the component of the preference between $i$ and $j$ that the transitive scale cannot explain. The decomposition of an observed preference matrix into a transitive gradient term and an intransitive residual is the subject of a combinatorial Hodge theory, and Bayesian and low-rank priors have been proposed for the array $\{\gamma_{ij}\}$ to control overfitting \citep{chenjoachims2016}. In an assessment context a material intransitive component is a warning that the unidimensional scale, and hence the rank order, is an inadequate summary of the judgements.

\subsection{Covariates and comparison-level effects}

It is often natural to explain the quality parameters in terms of object covariates, or to model features of the comparison itself. In the Bradley--Terry regression framework we write the quality parameter as a linear function of covariates and admit a comparison-level effect such as an order or presentation bias. This is given by
\begin{equation}
\logit \Prob(i \succ j) = \alpha + (\bx_i-\bx_j)^\top\bbeta,
\label{eq:bt-regression}
\end{equation}
where $\bx_i$ is a vector of covariates for object $i$, $\bbeta$ is the corresponding vector of coefficients, and $\alpha$ is an order effect that captures any systematic advantage accruing to, for instance, the object presented first \citep{davidsonbeaver1977,turnerfirth2012}. Such terms are directly relevant to the validity of CJ, since a nonzero $\alpha$ indicates that assessors are influenced by construct-irrelevant features of the presentation.

Table~\ref{tab:models} summarises the models and their relationships.

\begin{table}[t]
\centering
\small
\caption{Summary of the principal models for comparative judgement. Each is characterised by its outcome space and by the probability of the event that object $i$ is preferred to object $j$. All reduce to Bradley--Terry under the stated special case.}
\label{tab:models}
\begin{tabular}{@{}p{3.0cm}p{2.3cm}p{6.0cm}p{3.0cm}@{}}
\toprule
Model & Outcome & $\Prob(i \succ j)$ & Reduces to BT when \\
\midrule
Thurstone (Case V) & win / loss & $\Phi(\lambda_i-\lambda_j)$ & link set to logistic \\
Bradley--Terry & win / loss & $\sigma(\lambda_i-\lambda_j)$ & --- \\
Bradley--Terry--Luce & choice from set & $\pi_i/\sum_{k}\pi_k$ & set size two \\
Davidson (ties) & win / loss / tie & $\pi_i/(\pi_i+\pi_j+\nu\sqrt{\pi_i\pi_j})$ & $\nu=0$ \\
Plackett--Luce & ranking & sequential choice, Eq.~\eqref{eq:plackett-luce} & two objects ranked \\
Rater mixture & win / loss & $q_r\sigma(\lambda_i-\lambda_j)+(1-q_r)/2$ & $q_r=1$ for all $r$ \\
Intransitive & win / loss & $\sigma(\lambda_i-\lambda_j+\gamma_{ij})$ & $\gamma_{ij}=0$ \\
BT regression & win / loss & $\sigma(\alpha+(\bx_i-\bx_j)^\top\bbeta)$ & $\alpha=0$, saturated $\bbeta$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Estimation}
\label{sec:estimation}

The choice of estimation method is not separable from the choice of model, because the pathologies that motivate richer models, sparse designs and extreme objects in particular, also determine which estimator is well behaved. We distinguish three approaches.

Maximum likelihood is the default. The Bradley--Terry log-likelihood is concave in $\blambda$ and is readily maximised by iteratively reweighted least squares or Newton's method, and the same machinery extends to the regression form \eqref{eq:bt-regression} \citep{turnerfirth2012}. The difficulty is that the estimate diverges for any object that wins or loses all of its comparisons, an event that is common in CJ when the number of comparisons per object is small. In consequence the unmodified maximum likelihood estimator is often unsuitable for the sparse, adaptively selected designs that are typical of CJ.

Penalised estimation resolves this divergence and improves predictive accuracy. The bias-reducing penalty of \citet{firth1993} guarantees finite estimates, and a ridge penalty does so while also shrinking the scale towards equality. \citet{varinfirth2024} add a penalty of the form $\tfrac{1}{2\eta^2}\sum_i\lambda_i^2$ to the log-likelihood and tune the parameter $\eta$ by an empirical Bayes argument that avoids the need to refit the model, as would be required by cross-validation. Their approach yields appreciably better predictive accuracy than ordinary maximum likelihood, which is the relevant criterion when the fitted scale is to be used for prediction or for decisions.

Bayesian estimation places a prior on the quality parameters and bases inference on their posterior. A hierarchical prior of the form $\lambda_i \sim \mathrm{N}(0,\tau^2)$, with a hyperprior on $\tau$, shrinks the estimates and resolves the divergence problem automatically, since the prior supplies the regularisation that the likelihood lacks for extreme objects \citep{whelan2017}. Posterior computation proceeds by Markov chain Monte Carlo, and modern implementations use Hamiltonian Monte Carlo with the no-U-turn sampler, or bespoke data-augmentation schemes for the tie and ranking models \citep{forcedmarriage2022,bpcs2021}. Beyond resolving the divergence, the Bayesian treatment delivers honest interval estimates for derived quantities such as rank positions, which the plug-in approach tends to understate.

\section{Model comparison techniques}
\label{sec:comparison}

We now turn to the techniques for deciding between competing models. They fall into five groups, and a defensible analysis triangulates across them rather than relying on any single number. Table~\ref{tab:techniques} collects the techniques and the question each is suited to answer.

\subsection{Likelihood-based information criteria}

For models that are nested, a likelihood-ratio test is the natural starting point. We compare twice the difference in maximised log-likelihoods to a chi-squared distribution,
\begin{equation}
2\big(\ell_1-\ell_0\big) \sim \chi^2_{d_1-d_0},
\label{eq:lrt}
\end{equation}
where $\ell_1$ and $\ell_0$ are the maximised log-likelihoods of the larger and smaller models and $d_1-d_0$ is the difference in the number of free parameters. The Davidson model against Bradley--Terry, or the regression form against a constant order effect, are naturally compared in this way.

For non-nested models the standard tools are the Akaike and Bayesian information criteria,
\begin{equation}
\mathrm{AIC} = -2\ell + 2k, \qquad \mathrm{BIC} = -2\ell + k\log n,
\label{eq:aic-bic}
\end{equation}
where $\ell$ is the maximised log-likelihood, $k$ is the number of free parameters, and $n$ is the number of comparisons. The two criteria answer different questions, and the distinction is material in the CJ setting. The AIC is an estimate of out-of-sample predictive performance, derived as an approximation to the expected Kullback--Leibler divergence between the fitted and the true model, and it is the appropriate criterion when the scale is to be used for prediction. The BIC is an approximation to a Bayes factor and is consistent for identifying the true model when one of the candidates is true, a condition that is rarely plausible for human judgement data. Because a CJ scale is almost always built to support a decision, the predictive emphasis of the AIC is usually the more defensible, although neither criterion is reliable when the number of comparisons per object is small.

\subsection{Bayesian model comparison}

The Bayesian counterpart compares models through the marginal likelihood. The Bayes factor for model $M_1$ against $M_0$ is
\begin{equation}
B_{10} = \frac{p(D\given M_1)}{p(D\given M_0)} = \frac{\int p(D\given\theta_1,M_1)\,p(\theta_1\given M_1)\,d\theta_1}{\int p(D\given\theta_0,M_0)\,p(\theta_0\given M_0)\,d\theta_0},
\label{eq:bayes-factor}
\end{equation}
where $D$ denotes the data and $\theta_m$ the parameters of model $M_m$. The marginal likelihoods in \eqref{eq:bayes-factor} are integrals over the whole parameter space and are not in general tractable, so in practice they are estimated by bridge or importance sampling, or approximated through the BIC. The Bayes factor is sensitive to the choice of prior, and this sensitivity is a genuine drawback for the quality parameters, whose prior scale is not easy to specify a priori.

For this reason the predictive information criteria are often preferred within the Bayesian workflow. The widely applicable information criterion and the Pareto-smoothed importance sampling approximation to leave-one-out cross-validation both estimate out-of-sample predictive accuracy from the pointwise log-likelihood evaluated at the posterior draws, and both avoid the prior sensitivity and the intractable normalising constant of the Bayes factor \citep{bpcs2021}. They are computed by treating each comparison as a held-out point and accumulating its predictive density, and they are the most practical Bayesian tools for comparing CJ models.

\subsection{Out-of-sample predictive scoring}

The most direct and the most model-agnostic approach is to score the competing models on their ability to predict held-out comparisons. We partition the comparisons into a training and a test set, fit each model to the training set, and evaluate the predicted probabilities on the test set. A natural score is the mean logarithmic loss,
\begin{equation}
\mathrm{LL} = -\frac{1}{|T|}\sum_{c\in T}\Big[y_c\log\hat p_c + (1-y_c)\log(1-\hat p_c)\Big],
\label{eq:logloss}
\end{equation}
where $T$ is the test set, $y_c$ is the observed outcome of comparison $c$, and $\hat p_c$ is the probability the fitted model assigns to that outcome. The logarithmic loss rewards well-calibrated probabilities and penalises confident errors, and it is preferable to a simple classification accuracy, which ignores calibration. Repeating the partition in a cross-validation scheme reduces the dependence on any single split. This approach has the advantage of comparing models on exactly the quantity of interest, namely how well they capture the judging process, and it requires no assumption that any candidate model is true.

\subsection{Reliability-based measures}

The house metric of the CJ literature is the scale separation reliability, imported from Rasch measurement. It expresses the proportion of the observed variance in the estimated quality parameters that is attributable to true differences between objects rather than to estimation error. We write it as
\begin{equation}
\mathrm{SSR} = \frac{\hat\sigma^2_\lambda - \overline{\mathrm{SE}}^2}{\hat\sigma^2_\lambda},
\label{eq:ssr}
\end{equation}
where $\hat\sigma^2_\lambda$ is the observed variance of the estimated quality parameters and $\overline{\mathrm{SE}}^2$ is the mean squared standard error of those estimates. The SSR is useful as a descriptive summary, but it is a poor instrument for choosing between models, and it must be handled with care. \citet{bramley2015} showed by simulation that adaptive pair selection can inflate the SSR even for random data, and \citet{crompvoets2022} showed that it overestimates reliability when raters use different underlying constructs or when the true parameter variance is small. The SSR is in part an artefact of the design and the selection algorithm rather than a property of the model fit, and so it should not be used as the primary criterion for model comparison. A meta-analysis of reported CJ studies found that reaching an SSR of $0.70$ required on average at least fourteen comparisons per object, rising to thirty-seven for an SSR of $0.90$, which gives a sense of the design effort the statistic demands \citep{verhavert2019meta}.

A cleaner reliability check, and one that is closer to an external validation, is split-half or replicate reliability. Two independent panels of assessors judge the same set of objects, and the two resulting rank orders are correlated. A high correlation indicates that the scale is reproducible across panels, and a low correlation is direct evidence that the conclusions are unstable. This design has revealed both reassuring and troubling results in the literature, with high agreement in some mathematics studies and markedly lower agreement in an essay study, which is precisely the kind of instability that motivates the present question \citep{bramley2015,jonesalcock2014}.

\subsection{Rank-correlation and decision-level robustness}

The technique that bears most directly on the central question is the simplest. We fit each competing model to the same data and compare the resulting scales, both through rank correlation and, more importantly, through agreement on the decisions that the scale is used to make. The Spearman correlation between the estimated quality parameters under two models measures the stability of the full rank order, and a sensitivity analysis of this kind has been used to assess the influence of individual assessors as well as of modelling choices \citep{forcedmarriage2022}. Rank correlation alone, however, can mask disagreement at the boundaries that matter. We therefore recommend computing, in addition, the agreement on the operational decisions, for example the set of objects placed above a grade boundary or the identity of the top $k$ objects. Two models can correlate at $0.98$ over the full scale and still disagree on a material fraction of the borderline cases, and it is the borderline cases that determine whether a CJ study's conclusions are robust in any sense that matters to a candidate.

Finally, within any single model, fit can be assessed through the Rasch-style infit and outfit statistics, which flag objects or assessors whose observed outcomes are more or less predictable than the model expects. Values within roughly $0.5$ to $1.5$ are conventionally regarded as acceptable, and values outside this range identify misfitting assessors who may warrant exclusion or further scrutiny \citep{linacre2002}.

\begin{table}[t]
\centering
\small
\caption{Model comparison techniques, the question each addresses, and the chief caveat in the comparative judgement setting.}
\label{tab:techniques}
\begin{tabular}{@{}p{3.4cm}p{5.2cm}p{5.2cm}@{}}
\toprule
Technique & Question addressed & Chief caveat \\
\midrule
Likelihood-ratio test & Is the larger nested model justified? & Nested models only \\
AIC & Which model predicts best out of sample? & Unreliable for small designs \\
BIC / Bayes factor & Which model is most probable a posteriori? & Assumes a true model; prior sensitive \\
WAIC / LOO-CV & Bayesian out-of-sample accuracy & Needs pointwise log-likelihood \\
Predictive log loss & Direct accuracy on held-out comparisons & Choice of partition \\
Scale separation reliability & How reproducible is the scale? & Inflated by adaptivity; not a fit measure \\
Split-half reliability & Do independent panels agree? & Requires a replicate design \\
Rank correlation & Is the rank order stable across models? & Masks boundary disagreement \\
Decision-level agreement & Do the operational decisions change? & Requires a defined decision rule \\
Infit / outfit & Which objects or raters misfit? & Within-model diagnostic only \\
\bottomrule
\end{tabular}
\end{table}

\section{Implementation in practice}
\label{sec:practice}

The techniques above are well supported by existing software, and a complete analysis can be assembled without bespoke code. For maximum likelihood and penalised estimation of Bradley--Terry and its regression extensions, the \texttt{BradleyTerry2} package provides model fitting, quasi-variance standard errors, and the comparison intervals that do not depend on the identifiability constraint \citep{turnerfirth2012}. The \texttt{prefmod} package fits the wider class of pattern models, including order effects and subject-specific covariates \citep{hatzingerdittrich2012}. Ranking data and ties are handled by the \texttt{PlackettLuce} package, which adds pseudo-comparisons to guarantee finite estimates and supplies quasi standard errors \citep{turner2020plackettluce}. Bayesian fitting of Bradley--Terry, the Davidson tie model, and order-effect variants, together with the WAIC and leave-one-out comparison machinery, is provided by the \texttt{bpcs} package, which is built on Stan \citep{bpcs2021}. For the intransitive and rater-heterogeneity models a short bespoke Stan programme is generally required, since these are not yet packaged, but the additions to the linear predictor in \eqref{eq:rater-mixture} and \eqref{eq:intransitive} are straightforward to code.

We recommend the workflow set out in Algorithm~\ref{alg:workflow}. Its guiding principle is that the analysis should establish not whether a single model fits, but whether the substantive conclusions survive a deliberate attempt to overturn them by fitting plausible alternatives.

\begin{algorithm}[t]
\caption{A practical workflow for model comparison in a comparative judgement study}
\label{alg:workflow}
\begin{algorithmic}[1]
\State Fit a slate of competing models to the same comparisons: Bradley--Terry, Thurstone--Mosteller, the Davidson tie model if ties were recorded, a hierarchical Bayesian Bradley--Terry, and a rater-heterogeneity model.
\State Assess relative fit by predictive log loss on held-out comparisons, Eq.~\eqref{eq:logloss}, supported by AIC for the likelihood fits and by WAIC or leave-one-out cross-validation for the Bayesian fits.
\State Assess conclusion stability by the Spearman correlation between the estimated quality parameters under each pair of models.
\State Assess decision stability by the agreement on the operational decisions, such as the objects above each grade boundary or the top $k$, since these can disagree even when the rank correlation is high.
\State Anchor the analysis externally with a split-half replication across two assessor panels where the design permits.
\State Diagnose within the preferred model using infit and outfit statistics and the order effect $\alpha$, and report any evidence of construct-irrelevant influence or intransitivity.
\end{algorithmic}
\end{algorithm}

Two points of practical guidance follow from the literature. First, if the only competing models under consideration differ in their link function, that is Bradley--Terry against Thurstone--Mosteller, the analysis will almost certainly find the conclusions to be stable, and this stability is itself a reportable result rather than a reason to omit the comparison. The instructive divergences arise instead from structural assumptions, namely the treatment of ties, the presence of intransitivity, the heterogeneity of raters, and the bias induced by adaptive selection. Second, the standard reliability statistic should be reported for continuity with the literature but should not carry the weight of the model comparison, for the reasons given in Section~\ref{sec:comparison}. The predictive and decision-level criteria are the ones that answer the question actually being asked.

\section{Discussion}
\label{sec:discussion}

We have summarised the principal models for comparative judgement within a single framework, described the machinery for comparing them, and proposed a workflow for applying that machinery to a real study. The recurring theme is that robustness is not a single property. Conclusions are robust to the choice of link function, since the Bradley--Terry and Thurstone models are near-indistinguishable in practice, but they can be sensitive to the structural assumptions that govern ties, intransitivity, rater heterogeneity, and the selection of pairs. A CJ analysis that fits one model and reports one reliability coefficient cannot detect this sensitivity, and may present as settled a conclusion that a plausible alternative model would unsettle.

The framework has limitations that suggest natural extensions. We have treated the comparison data as given, but the design and the analysis interact, and the adaptive selection of pairs that inflates the scale separation reliability also complicates out-of-sample prediction, since the held-out comparisons are not a random sample of the possible comparisons. A model comparison that accounts for the selection mechanism, rather than conditioning on the realised design, would be a worthwhile development. We have also said little about the construct itself. A material intransitive component or a strongly nonzero order effect is evidence that the unidimensional scale is an inadequate representation of the judgements, and in that case the appropriate response is not to choose a better one-dimensional model but to question whether the construct supports a single scale at all. The tools assembled here will identify when that question needs to be asked, which is a precondition for answering it.

\begin{thebibliography}{99}
\small

\bibitem[Agresti(2002)]{agresti2002}
Agresti, A. (2002). \textit{Categorical Data Analysis}, 2nd edn. Wiley, New York.

\bibitem[Bradley and Terry(1952)]{bradleyterry1952}
Bradley, R.~A. and Terry, M.~E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. \textit{Biometrika}, 39(3/4), 324--345.

\bibitem[Bramley(2015)]{bramley2015}
Bramley, T. (2015). \textit{Investigating the Reliability of Adaptive Comparative Judgment}. Cambridge Assessment Research Report, Cambridge.

\bibitem[Bramley and Vitello(2018)]{bramleyvitello2018}
Bramley, T. and Vitello, S. (2018). The effect of adaptivity on the reliability coefficient in adaptive comparative judgement. \textit{Assessment in Education: Principles, Policy \& Practice}, 26(1), 43--58.

\bibitem[Crompvoets et al.(2022)]{crompvoets2022}
Crompvoets, E.~A.~V., B\'eguin, A.~A. and Sijtsma, K. (2022). On the bias and stability of the results of comparative judgment. \textit{Frontiers in Education}, 6, 788202.

\bibitem[Davidson(1970)]{davidson1970}
Davidson, R.~R. (1970). On extending the Bradley--Terry model to accommodate ties in paired comparison experiments. \textit{Journal of the American Statistical Association}, 65(329), 317--328.

\bibitem[Davidson and Beaver(1977)]{davidsonbeaver1977}
Davidson, R.~R. and Beaver, R.~J. (1977). On extending the Bradley--Terry model to incorporate within-pair order effects. \textit{Biometrics}, 33(4), 693--702.

\bibitem[Dittrich et al.(1998)]{dittrich1998}
Dittrich, R., Hatzinger, R. and Katzenbeisser, W. (1998). Modelling the effect of subject-specific covariates in paired comparison studies with an application to university rankings. \textit{Journal of the Royal Statistical Society: Series C}, 47(4), 511--525.

\bibitem[Firth(1993)]{firth1993}
Firth, D. (1993). Bias reduction of maximum likelihood estimates. \textit{Biometrika}, 80(1), 27--38.

\bibitem[Hatzinger and Dittrich(2012)]{hatzingerdittrich2012}
Hatzinger, R. and Dittrich, R. (2012). prefmod: An R package for modeling preferences based on paired comparisons, rankings, or ratings. \textit{Journal of Statistical Software}, 48(10), 1--31.

\bibitem[Jones and Alcock(2014)]{jonesalcock2014}
Jones, I. and Alcock, L. (2014). Peer assessment without assessment criteria. \textit{Studies in Higher Education}, 39(10), 1774--1787.

\bibitem[Linacre(2002)]{linacre2002}
Linacre, J.~M. (2002). What do infit and outfit, mean-square and standardized mean? \textit{Rasch Measurement Transactions}, 16(2), 878.

\bibitem[Luce(1959)]{luce1959}
Luce, R.~D. (1959). \textit{Individual Choice Behavior: A Theoretical Analysis}. Wiley, New York.

\bibitem[Mosteller(1951)]{mosteller1951}
Mosteller, F. (1951). Remarks on the method of paired comparisons: I. \textit{Psychometrika}, 16(1), 3--9.

\bibitem[Phelan and Whelan(2017)]{whelan2017}
Whelan, J.~T. (2017). Prior distributions for the Bradley--Terry model of paired comparisons. arXiv:1712.05311.

\bibitem[Plackett(1975)]{plackett1975}
Plackett, R.~L. (1975). The analysis of permutations. \textit{Journal of the Royal Statistical Society: Series C}, 24(2), 193--202.

\bibitem[Pollitt(2012)]{pollitt2012}
Pollitt, A. (2012). The method of adaptive comparative judgement. \textit{Assessment in Education: Principles, Policy \& Practice}, 19(3), 281--300.

\bibitem[Chen and Joachims(2016)]{chenjoachims2016}
Chen, S. and Joachims, T. (2016). Modeling intransitivity in matchup and comparison data. In \textit{Proceedings of the Ninth ACM International Conference on Web Search and Data Mining}, 227--236.

\bibitem[Thurstone(1927)]{thurstone1927}
Thurstone, L.~L. (1927). A law of comparative judgment. \textit{Psychological Review}, 34(4), 273--286.

\bibitem[Turner and Firth(2012)]{turnerfirth2012}
Turner, H. and Firth, D. (2012). Bradley--Terry models in R: the BradleyTerry2 package. \textit{Journal of Statistical Software}, 48(9), 1--21.

\bibitem[Turner et al.(2020)]{turner2020plackettluce}
Turner, H.~L., van Etten, J., Firth, D. and Kosmidis, I. (2020). Modelling rankings in R: the PlackettLuce package. \textit{Computational Statistics}, 35(3), 1027--1057.

\bibitem[Varin and Firth(2024)]{varinfirth2024}
Varin, C. and Firth, D. (2024). Tractable ridge regression for paired comparisons. arXiv:2406.09597.

\bibitem[Verhavert et al.(2018)]{verhavert2018ssr}
Verhavert, S., De Maeyer, S., Donche, V. and Coertjens, L. (2018). Scale separation reliability: what does it mean in the context of comparative judgment? \textit{Applied Psychological Measurement}, 42(6), 428--445.

\bibitem[Verhavert et al.(2019)]{verhavert2019meta}
Verhavert, S., Bouwer, R., Donche, V. and De Maeyer, S. (2019). A meta-analysis on the reliability of comparative judgement. \textit{Assessment in Education: Principles, Policy \& Practice}, 26(5), 541--562.

\bibitem[bpcs(2021)]{bpcs2021}
Issa Mattos, D. and Martins Silva Ramos, \'E. (2021). Bayesian paired-comparison with the bpcs package. \textit{Behavior Research Methods}, 54, 2025--2045.

\bibitem[Spearing et al.(2022)]{forcedmarriage2022}
Spearing, H., Tawn, J.~A., Irons, D.~J., Paine, T. and Bocci, C. (2022). Comparative judgement modelling to map forced marriage at local levels. arXiv:2212.01202.

\end{thebibliography}

\end{document}
