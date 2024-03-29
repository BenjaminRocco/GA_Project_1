
# Establishing a Linear Correlation Between SAT Participation and Scores

## Objective

Problem Statement: How may levels of preparedness be assessed in test takers given increased participation levels? The goal of this project is to analyze trends in SAT scores among test takers from $2017$ to $2019.$ Additionally, we will observe trends in $2019$ scores among intended majors and perform statistical analysis methods to to analyze various features of these scores.

Our hypothesis will be, in general, that although participation levels increased in the SATs from $2017$ to $2019,$ test scores decreased slightly across these years in all areas. In other words, these two variables were negatively correlated and a linear regression line may be observed across these years. We also assert that those in intended majors relevant to test subjects performed better in those areas in $2019$, and those with non-academic intended majors, in general, performed under the average of $2019$ general SAT standards. 

## Data

**The first few files `SAT_2017_NEW, SAT_2018_NEW, SAT_2019_RESET_NEW` follow this format**

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**state**|*string*|$2017-2019$ State SAT Scores|States of exam takers shown in alphabetical order| 
|**participation**|*float*|$2017-2019$ State SAT Scores|The percentage of exam takers within the state given in decimal format|
|**ebrw**|*integer*|$2017-2019$ State SAT Scores|Score in the Evidence-Based Reading and Writing section of the SAT|
|**math**|*integer*|$2017-2019$ State SAT Scores|Score in the Math section of the SAT|
|**total**|*integer*|$2017-2019$ State SAT Scores|Summation of the EBRW and Math sections of the SAT|

**Final file `SAT_2019_MAJ_NEW` follows following format**

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**intendedcollegemajor**|*string*|$2019$ State SAT Scores by Major|Various majors of exam takers shown in alphabetical order| 
|**testtakers**|*float*|$2019$ State SAT Scores by Major|The number of exam takers of the given intended major|
|**percent**|*float*|$2019$ State SAT Scores by Major|The percentage of exam takers of the intended major out of all majors given in decimal format|
|**total**|*integer*|$2019$ State SAT Scores by Major|Summation of the Reading & Writing and Math sections of the SAT|
|**readingwriting**|*integer*|$2019$ State SAT Scores by Major|Score in the Reading and Writing section of the SAT|
|**math**|*integer*|$2019$ State SAT Scores by Major|Score in the Math section of the SAT|

## Data Cleaning

The only data that was dropped were scores from test takers from Puerto Rico and the Virgin Islands; thus, for this study only the $50$ states were considered and The Dictrict of Columnbia (D.C.). 

## Work Section

**All work shown**

All work shown and significant figures $\bar{x} \pm \tilde{\sigma}$ shown, for mean $\bar{x}$ and standard error $\tilde{\sigma}$.

## Visuals

**Analysis section**

We utilize figures to assert key mathematical relationships regarding our variables, with the common connecting thread being that with increased participation there is a subsequent decrease in test scores. 

First, we exhibit a barchart showing increased participation among 20 states with highest participation levels.

Second, we exhibit a box-and-Whisker plot showing total mean scores decreased slightly over the course of the three years $(2017-2019).$

Afterwards, we feature scatter plots accentuating the linear regression line that accurately captures the negative correlation between participation levels and total mean scores from $2017$ to $2019.$

Bar charts showing anticipated trend among intended majors in participation and total mean scores. 

## Conclusion

**Recommendations**

We examined trends in SAT scores and participation rates from $2017$ through $2019$ and also looked at trends in intended major SAT scores and participation in $2019.$ We found that, although participation increased in these years, SAT scores declined slightly as a result of new test takers entering the pool, i.e. participation did not lead to preparedness and these variables were negatively correlated. 

Intended majors performed better in their areas of expertise compared with those not in that area. Those in the sciences performed better overall, most likely due to objectivity of mathematics questions allowing for clear-cut and pragmatic approaches to solutions, versus the multi-interpretability of reading and writing questions which could lead to greater variability in answers and subsequently more inccorect answers. Finally, cultural factors and disparities in tutoring and study resources that certain demographics have readily accessible to them are another cause of potential score decreases, as these groups are pressured to increase their exam participation but are not provided adequate reasources to prepare themselves for the academic rigors of the exam sufficiently. 

Sources: 

https://www.washingtonpost.com/local/education/sat-scores-drop-for-2019-class-but-participation-rises-through-testing-in-schools/2019/09/23/332fc4d0-de11-11e9-8dc8-498eabc129a0_story.html

https://www.brookings.edu/articles/sat-math-scores-mirror-and-maintain-racial-inequity/
