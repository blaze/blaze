# Documentation end-to-end examples, walkthroughs, quickstart 

## Example with remote live data: Yahoo Finance. <br/>Static data: US Gov public data

* Access with `http` URIs with `blaze.data`
  - Notebook example of [Yahoo Finance Data](yhoo_finance_example.ipynb)
  - Sample CSV data from `catalog.data.gov` for [`Impaired Driving In 2012`](impaired_driving_data_us_states_2012.ipynb)
    - Rendered plots on [anaconda.org](https://beta.anaconda.org/jsandhu/impaired_driving_data_us_states_2012/notebook)


* Describe briefly download and caching with `Temp(CSV)`
  In both examples above:

  - data is downloaded to a temp location and queried to present results during interactive exploration
  - downloading and managing of CSV file is done automatically by blaze
  - user doesn't need to know where the data is physically downloaded

* Point out how it updates when re-run with latest data

