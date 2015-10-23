with recursive dates as (
  select
    min(aggs.pickup_datetime) as ts
  union all
  select
    dateadd(DAY, 1, ts) as ts
  from
    dates
  where ts < max(aggs.pickup_datetime)
)
