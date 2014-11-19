"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
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
            pmf = thinkbayes2.MakePoissonPmf(lt, 60)
            # thinkplot.Pdf(pmf, linewidth=1, alpha=.2, color='purple')
            metaPmf[pmf] = prob
        superMeta = thinkbayes2.MakeMixture(metaPmf)
        superMeta += score
        print(superMeta.ProbGreater(homerunAvg))
        print(superMeta.MaximumLikelihood())
        # thinkplot.Pdf(superMeta)

def homersSeasonCalculator(homerunHistory, gamesPlayed, currentHR):
    diffList = []
    weightedDelta = 0
    for i in range(len(homerunHistory)-1):
        delta = homerunHistory[i+1]-homerunHistory[i]
        weightedDelta += float(delta)*(i+1)/6

    print(weightedDelta)
    previousYearHR = homerunHistory[-1]
    homersSeasonBase = previousYearHR + weightedDelta
    seasonFraction = float(gamesPlayed)/162
    homersSeasonRate = currentHR / seasonFraction
    print(homersSeasonRate)
    compositeHomers = seasonFraction*homersSeasonRate + (1-seasonFraction) * homersSeasonBase

    return compositeHomers

def main():
    previousYears = [29, 28, 33, 27]
    gamesRemaining = 95
    homeRuns = 2
    homersSeason = homersSeasonCalculator(previousYears, (162-gamesRemaining), homeRuns)
    print(homersSeason)
    previousYears.append(homersSeason)
    hypos = numpy.linspace(0, homersSeason*2, 201)
    suite = Season(hypos)
    
    for year in previousYears:
        suite.Update(162/year)
    
    thinkplot.Pdf(suite, label='prior')

    suite.PredRemaining(gamesRemaining,homeRuns,homersSeason)


    thinkplot.Show()


if __name__ == '__main__':
    main()
