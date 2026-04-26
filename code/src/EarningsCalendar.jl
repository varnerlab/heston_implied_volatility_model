"""
    EarningsCalendar

Loader and accessors for the earnings calendar fetched by
`code/scripts/fetch_earnings_calendar.py`. Provides:

  - `load_earnings(path)`: read the CSV into a Dict{String,Vector{Date}}
    keyed by ticker.
  - `days_to_earnings(cal, ticker, date)`: signed integer — calendar
    days from `date` to the nearest earnings print (negative = past
    print, positive = upcoming). Returns `missing` if the ticker has
    no calendar entry (e.g. ETFs).
  - `in_earnings_window(cal, ticker, date; days)`: `Bool`, true iff the
    nearest print is within ±days calendar days of `date`.

Both options 2 (`days_to_earnings` as NN feature) and 3 (per-day theta
inside earnings windows) consume these accessors.
"""
module EarningsCalendar

using CSV
using DataFrames
using Dates

export EarningsCal, load_earnings, days_to_earnings, in_earnings_window,
       nearest_earnings

const EarningsCal = Dict{String,Vector{Date}}

"""
    load_earnings(path::AbstractString) -> EarningsCal

Read the earnings-calendar CSV produced by
`code/scripts/fetch_earnings_calendar.py` and return a dict mapping
ticker to a sorted vector of earnings `Date`s.
"""
function load_earnings(path::AbstractString)
    df = CSV.read(path, DataFrame)
    cal = EarningsCal()
    for row in eachrow(df)
        t = String(row.ticker)
        d = row.earnings_date isa Date ? row.earnings_date : Date(String(row.earnings_date))
        push!(get!(cal, t, Date[]), d)
    end
    for (t, v) in cal
        sort!(v)
    end
    return cal
end

"""
    nearest_earnings(cal, ticker, date) -> Union{Date,Missing}

Return the earnings date nearest to `date` for `ticker`, or `missing`
if the ticker has no calendar entry.
"""
function nearest_earnings(cal::EarningsCal, ticker::AbstractString, date::Date)
    dates = get(cal, String(ticker), nothing)
    (dates === nothing || isempty(dates)) && return missing
    diffs = abs.(Dates.value.(dates .- date))
    return dates[argmin(diffs)]
end

"""
    days_to_earnings(cal, ticker, date) -> Union{Int,Missing}

Signed calendar days from `date` to the nearest earnings print:
negative = past print, positive = upcoming, 0 = print day.
Returns `missing` for tickers with no calendar entry (e.g. ETFs).
"""
function days_to_earnings(cal::EarningsCal, ticker::AbstractString, date::Date)
    nxt = nearest_earnings(cal, ticker, date)
    nxt === missing && return missing
    return Dates.value(nxt - date)
end

"""
    in_earnings_window(cal, ticker, date; days::Int=3) -> Bool

`true` iff the nearest earnings print is within ±`days` calendar days
of `date`. Tickers with no calendar entry return `false` (treat as
"never in window").
"""
function in_earnings_window(cal::EarningsCal, ticker::AbstractString, date::Date;
                            days::Int=3)
    d = days_to_earnings(cal, ticker, date)
    d === missing && return false
    return abs(d) <= days
end

end # module
