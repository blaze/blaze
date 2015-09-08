with filled_dates as (
  select day, 0 as blank_count from
    generate_series('2014-01-01 00:00'::timestamptz,
                    current_date::timestamptz,
                    '1 day')
      as day
),
signup_counts as (
  select date_trunc('day', created_at) as day, count(*) as signups
    from users
  group by date_trunc('day', created_at)
)
select filled_dates.day,
       coalesce(signup_counts.signups, filled_dates.blank_count) as signups
  from filled_dates
    left outer join signup_counts on signup_counts.day = filled_dates.day
  order by filled_dates.day;
