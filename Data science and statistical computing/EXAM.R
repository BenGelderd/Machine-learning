install.packages("ggplot2")
install.packages("tidyverse")
install.packages("dplyr")
install.packages("shiny")
install.packages("lubridate")
install.packages("anytime")
install.packages("tidyr")

library("ggplot2")
library("tidyverse")
library("dplyr")
library("shiny")
library("lubridate")
library("anytime")

download.file("https://www.louisaslett.com/Courses/DSSC/computer_exam/hotels.csv", "hotels.csv")
download.file("https://www.louisaslett.com/Courses/DSSC/computer_exam/iso.csv", "iso.csv")

#Q11.1
hotels <- hotels |> rename(is_cancelled = is_canceled)

ggplot(hotels, aes(x = daily_rate, y = total_nights)) + geom_point()


#Q11.2
hotels2 <- hotels |> filter(daily_rate < 4000)

ggplot(hotels2, aes(x = daily_rate, y = total_nights)) + geom_hex() +
  ylab("Total Nights Stayed") + xlab("Daily Rate ($)")

#Q11.3
cancellations <- hotels |> group_by(country_code) |>
  summarise(total = sum(total_nights >= 0), 
            cancellations = sum(is_cancelled),
            percent_cancelled = (cancellations/total)*100)

iso <-  iso |> rename(country_code = code)
cancellations2 <- merge(cancellations, iso, by = "country_code") |>
  arrange(desc(total)) |> 
  relocate(country, .after = country_code)

min <- as.numeric(cancellations2 |> filter(country == "Netherlands") |> select(total))
  
ggplot(cancellations2 |> filter(total >= min)) +
  geom_col(aes(percent_cancelled, country), fill = "red") +
  xlab("Cancelled Bookings (%)") 

#Q11.4
hotels3 <- hotels |> mutate("kids" = case_when(children > 0 ~"Kids", 
                                                babies > 0 ~ "Kids",
                                                children == 0 ~ "No Kids",
                                                babies == 0~ "No Kids"))


hotels3 <- hotels3 |> mutate("dow" = weekdays(ydm(
  str_glue("{arrival_year}-{arrival_day}-{arrival_month}"))))

#Q11.5
total <- hotels3 |> group_by(kids) |> summarise(count = sum(total_nights >= 0), dow)

ggplot(total, aes(x = dow, y = count, fill = kids)) + geom_bar(stat="identity") + 
  facet_wrap()
