# Hedging LPI Liabilities

THIS REPO AND THIS DOCUMENT IS CURRENTLY WORK IN PROGRESS. 

This is a repo that aspires to help with hedging inflation for UK pension funds. 

Inflation linkage in pension funds is extremely complex. Linkages have different caps and floors over different periods of time and both year-on-year and over a period of years caps exist. And while legislation sets minimum increases for benefits in deferment or in payment, each scheme will have its own rules.

This study can be thought as being a story of 4 parts. 

In part 1 we start by setting up a model for calibrating inflation as the underlying variable to be projected over a future horizon (arbitrarily chosen to be 40 years for now). We used CPI as our chosen variable - although probably CPIH would have been a better choice, given that RPI will migrate to CPIH by 2030. This part is largely completed and a document is already available. 

The second part is to design an imaginary but sufficiently complex pension scheme whose liabilities we wish to hedge. That part is also largely done, and a fair amount of complexity is modelled; significantly more complexity may well exist in actual schemes - but we felt that its modelling might make this study hard to track - so we only focused on different type of linkages. That part is also largely complete (its probably the most trivial part of this work).

The third part is the identification of available instruments to help us with this exercise. This include inflation swaps and inflation options. At this part we need to either consider the pricing given, or to determine it based on the inflation parameters at the time of the exercise. Getting this part right has a large bearing on the next and final part. Currently, this part is work in progress. 

The final part is the hedging itself. Again, before we start we need to decide on our objectives. It is possible that different objectives result in a different optimal portfolio. This part should also look at the sensitivity of the chosen portfolio to plausible stresses. This part has not started yet. 

For the time being a number of complications are parked - these include timing of cashflows during the year (all are assumed to occur once a year, usually mid-year), reporting and indexation lags and rule changes from RPI to CPI. Our focus remains the best hedging of inflations with different linkage definitions as well as varying caps and floors. 

Any comments/suggestions, please email nikoskatrakis@gmail.com
