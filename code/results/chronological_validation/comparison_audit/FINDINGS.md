# Follow-up audit of the forecast comparison

The author's question prompted this audit after the first manuscript package was
built. These findings have not yet been incorporated into that package. Existing
experiment scores and protocols were preserved.

All 64 shorter-maturity five-session outcomes were checked against the selected
raw source files at both origin and endpoint. OCC symbol, ticker, put/call,
strike, expiration, underlying session date, origin stock price and IV, origin
midpoint, endpoint bid/ask and midpoint, and absolute-error arithmetic matched.
Prices were dollars per share throughout. The observed-stock comparison matched
the same contracts, origins, endpoints and observed prices: 26 GS and 24 LLY
outcomes, each covering 13 origins. It supplies future stock information and is
only a conditional diagnostic, not an equally informed forecast competitor.

The runner calls JumpHMM.simulate without its `start` argument. The installed
package defaults to `start=:stationary`, sampling each initial state from the
long-run distribution. Recent observed returns do not update the state at an
origin. Updating the stock price and option IV therefore does not make this a
forecast conditioned on recent return history. Both saved marginals use dt=1/252
and rf=0. The independent pilot imposes average annual log growth of 10%, about
0.2% over five sessions. The shift cannot itself explain several-percent misses.

The original primary metric was MAE of the forecast mean. That is well-defined
but omits a relevant distinction for skewed option-price distributions: median
forecasts minimize expected absolute loss; means minimize expected squared loss.
Without new simulations, median-based five-session MAE is 2.480831 for coupled GS
and 2.485551 for frozen GS, versus mean-based MAE 4.200236 and 4.220576. For LLY,
median-based errors are 11.791973 and 11.778834, versus mean-based errors 11.587747
and 11.620663. Report both choices transparently rather than replacing the
prespecified mean-based result after seeing outcomes. All variants are included
in point_summary.csv. This does not establish a coupled-factor advantage.

For a concrete LLY example, the August 4 origin stock value was 1117.47 and the
August 11 observed value was 1212.92, an 8.54% rise. The stock forecast mean was
1118.87, with a central 90% interval of 1051.97 to 1190.78. The same August 21
1175-strike call had an endpoint quote midpoint of 48.985 versus forecast mean
20.440786. Other large LLY errors coincide with subsequent large declines as
well as rises. This identifies actual misses; it does not establish that those
moves could have been predicted from origin information.

Data comparability is not completely verified. The SDK's stock bars default to
IEX; the options snapshot call does not specify a feed. Alpaca documents a
default indicative options feed for accounts without a subscription, with
modified quotes. Given the author's stated free tier, indicative data is the
expected feed, but the historical files do not record the feed or entitlements.
They also omit the option quote timestamps. Capture times and daily bar session
labels cannot prove stock/option synchronization or quote freshness. This is a
limitation, not a demonstrated explanation of the dollar errors. Endpoint-IV
repricing agreement is likewise not independent verification of feed accuracy.

Sources checked September 7, 2026:
- https://otexts.com/fpp3/accuracy.html
- https://docs.alpaca.markets/us/reference/optionsnapshots

The initial package should be revised to distinguish point summaries, disclose
stationary state initialization and feed limitations, and avoid interpreting the
conditional diagnostic as validation of an operational forward forecast. A
causal state-initialization comparison would be a separate, documented experiment.
