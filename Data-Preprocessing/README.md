---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
# Data Cleaning - Preprocessing
Text Data Preprocessing For Topic Modeling
\begin{algorithm}
	\caption{Data Cleaning}
	\label{dc}
	\begin{algorithmic}[1]
		\Procedure{DataCleaning}{$CORPUS$}
		%\State $stripped\_text$ = LowerCase($CORPUS$)
		\State $stripped\_text$ = Lower case all text from $CORPUS$
		\State $stripped\_text$ = Remove $\{\# \,\, n’t\,\,  !\,\,   @\,\,  \,\, ,  “\,\,   "\,\,  ’s\,\,  ()\,\,’ \,\, ? \,\, \}$ from $stripped\_text$
		\State $stripped\_text$ = Remove one-digit and non-digit characters from $stripped\_text$
		\State $stripped\_text$ = Remove mail server and domain of the existing email addresses from $stripped\_text$
		\State $stripped\_text$ = Remove URLs from $stripped\_text$
		\State 

		\For {each $line$ in $stripped\_text$}
		    \If {$line$ starts with expression 'rt'}
		        \State $stripped\_text$ = Remove 'rt' from $stripped\_text$
		    \EndIf
		\EndFor
		
		\For {each $token$ in $stripped\_text$}
            \State $token$ = Lemmatize($token$)
            \State $token$ = Stemmed($token$)
		\EndFor
		
		\State Return $stripped\_text$
		\EndProcedure
    \end{algorithmic}
\end{algorithm}
