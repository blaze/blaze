with aggs as (
  select
    date_trunc('day', pickup_datetime) as pickup_datetime,
    sum(passenger_count) as passenger_count_sum,
    avg(fare_amount) as fare_amount_mean
  from
    samp
  group by
    date_trunc('day', pickup_datetime)
), ts as (
  select generate_series(
      min(aggs.pickup_datetime),
      max(aggs.pickup_datetime),
      interval '1 day'
    ) as pickup_datetime,
    0 as passenger_count_sum,
    cast(null as numeric) as fare_amount_mean
  from
    aggs
), result as (
  select
    ts.pickup_datetime as stamp,
    coalesce(aggs.passenger_count_sum, ts.passenger_count_sum) as passenger_count_sum,
    coalesce(aggs.fare_amount_mean, ts.fare_amount_mean) as fare_amount_mean
  from
    ts
      left outer join
    aggs
      on ts.pickup_datetime = aggs.pickup_datetime
)
select
  stamp as pickup_datetime,
  passenger_count_sum,
  fare_amount_mean
from result
order by pickup_datetime;
