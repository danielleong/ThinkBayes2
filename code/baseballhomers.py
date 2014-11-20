"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

The Robinson Cano Problem
This code seeks to calculate rest of season Home runs based on career and season totals
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot

class Season(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis."""

        x = data
        lam = hypo/162
        like = thinkbayes2.EvalExponentialPdf(x,lam)

        return like

    def PredRemaining(self, rem_games, score, homerunAvg):
        """Plots the predictive distribution for final number of goals.

        rem_time: remaining time in the game in minutes
        score: number of goals already scored
        """
        metaPmf = thinkbayes2.Pmf()
        gameFraction = rem_games/162
        for lam, prob in self.Items():
            lt = lam*gameFraction
            pmf = thinkbayes2.MakePoissonPmf(lt, int(homerunAvg*2))
            metaPmf[pmf] = prob
        superMeta = thinkbayes2.MakeMixture(metaPmf)
        superMeta += score
        totalHR = superMeta.MaximumLikelihood()
        thinkplot.Pdf(superMeta, label="Predictive Distribution")
        return totalHR

def homersSeasonCalculator(homerunHistory, gamesPlayed, currentHR):
    """" Given previous four years, number of games played in current season,
        and current HR total will calculate the rate of home runs for rest of 
        year"""

    weightedDelta = 0

    for i in range(len(homerunHistory)-1):
        delta = homerunHistory[i+1]-homerunHistory[i]
        weightedDelta += float(delta)*(i+1)/6

    previousYearHR = homerunHistory[-1]
    homersSeasonBase = previousYearHR + weightedDelta
    seasonFraction = float(gamesPlayed)/162
    homersSeasonRate = currentHR / seasonFraction

    compositeHomers = seasonFraction*homersSeasonRate + (1-seasonFraction) * homersSeasonBase

    return compositeHomers

def main():
    """ Main loop creates a vector, updates it with home run totals and predicts
        number of home runs that Hitter will hit during season.
        previousYears = Cano's previous four home run totals
        gamesRemaining = Number of games left in 162 Game seasonFraction
        homeRuns: Home runs at point in season evaluated"""

    previousYears = [29, 28, 33, 27]
    gamesRemaining = 95
    homeRuns = 2

    homersSeason = homersSeasonCalculator(previousYears, (162-gamesRemaining), homeRuns)
    print(homersSeason)
    previousYears.append(homersSeason)
    hypos = numpy.linspace(0, homersSeason*2.25, 201)
    suite = Season(hypos)
    
    for year in previousYears:
        suite.Update(162/year)
    
    # thinkplot.Pdf(suite, label='prior')

    totalHR = suite.PredRemaining(gamesRemaining,homeRuns,homersSeason)
    print('Robinson Cano HR total 2014: ' + str(totalHR))

    # thinkplot.Show(xlabel="Home Runs", ylabel="Pmf", title='Prior Distribtion of HRs')
    thinkplot.Save(root = 'BaseballPosterior', xlabel="Home Runs", ylabel="Pmf", title='Distribution of Cano 2014 Home Run Total', formats=['png'])

if __name__ == '__main__':
    main()
