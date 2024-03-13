
# Establishing a Linear Correlation Between SAT Participation and Scores Three Years Prior to the Coronavirus Pandemic

## Objective

The goal of this project is to analyze trends in SAT scores among test takers from 2017 to 2019. Our hypothesis will be, in general, that although participation levels increased in the SATs from 2017 to 2019, test scores decreased slightly across these years in all areas. In other words, these two variables were negatively correlated and a linear regression line may be observed across these years. Additionally, we will observe trends in 2019 scores among intended majors and perform statistical analysis methods to to analyze various features of these scores, asserting that those in intended majors relevant to test subjects performed better in those areas, and those with non-academic intended majors, in general, performed under the average of 2019 general SAT standards. 

## Data

**First few files `SAT_2017_NEW, SAT_2018_NEW, SAT_2019_RESET_NEW` Follow this format**

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**state**|*string*|2017-2019 State SAT Scores|States of exam takers shown in alphabetical order| 
|**participation**|*float*|2017-2019 State SAT Scores|The percentage of exam takers within the state given in decimal format|
|**ebrw**|*integer*|2017-2019 State SAT Scores|Score in the Evidence-Based Reading and Writing section of the SAT|
|**math**|*integer*|2017-2019 State SAT Scores|Score in the Math section of the SAT|
|**total**|*integer*|2017-2019 State SAT Scores|Summation of the EBRW and Math sections of the SAT|

**Final file `SAT_2019_MAJ_NEW` follows following format**

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**intendedcollegemajor**|*string*|2019 State SAT Scores by Major|Various majors of exam takers shown in alphabetical order| 
|**testtakers**|*float*|2019 State SAT Scores by Major|The number of exam takers of the given intended major|
|**percent**|*float*|2019 State SAT Scores by Major|The percentage of exam takers of the intended major out of all majors given in decimal format|
|**total**|*integer*|2019 State SAT Scores by Major|Summation of the Reading & Writing and Math sections of the SAT|
|**readingwriting**|*integer*|2019 State SAT Scores by Major|Score in the Reading and Writing section of the SAT|
|**math**|*integer*|2019 State SAT Scores by Major|Score in the Math section of the SAT|

## Work Section

**All work shown**

All work shown and significant figures $\bar{x} \pm \tilde{\sigma}$ shown, for mean $\bar{x}$ and standard error $\tilde{\sigma}$.

## Visuals

**Analysis section**

Barchart showing increased participation among 20 states with highest participation levels

Box-and-Whisker plot showing total mean scores decreased slightly over the course of the three years (2017-2019).

Scatter plot showing linear regression line accurately display negative correlation between participation levels and total mean scores over three years (2017-2019).

Bar charts showing anticipated trend among intended majors in participation and total mean scores. 

## Conclusion

**Recommendations**

We examined trends in SAT scores and participation rates from 2017 through 2019 and also looked at trends in intended major SAT scores and participation in 2019. We found that, although participation increased in these years, SAT scores declined slightly as a result of new test takers entering the pool, i.e. participation did not lead to preparedness and these variables were negatively correlated. 

Intended majors performed better in their areas of expertise compared with those not in that area. Those in the sciences performed better overall, most likely due to objectivity of mathematics questions / multi-interpretability of reading and writing questions, as well as cultural factors and disparities in resources. 
