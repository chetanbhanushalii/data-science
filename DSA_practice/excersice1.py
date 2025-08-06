#------------------------ Question 1 ------------------------
# nyc_weather.csv contains new york city weather for first few days in the month of January. 
# Write a program that can answer following,
# What was the average temperature in first week of Jan
# What was the maximum temperature in first 10 days of Jan
# Figure out data structure that is best for this problem
# Using a dictionary to store date as key and temperature as value for easy access

#read csv file using python
import csv
def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header
        data = [row for row in csv_reader]
    return header, data

data = read_csv('nyc_weather.csv')
# print(data)

date_data = data[1]
# print(type(date_data))

weather_data = {}

# Populate the dictionary with data from the CSV
for row in date_data:
    date = row[0]  # first column is date
    temperature = int(row[1])  # second column is temperature
    weather_data[date] = temperature

print(weather_data)

length_data = len(weather_data)

sum_temp = 0
count = 0
for date,temperature in weather_data.items():
    print(f"Date: {date}, Temperature: {temperature}")
    sum_temp += temperature
    count += 1
    if count == 7:  # First week
        break
average_temp = sum_temp / length_data
print(f"Average temperature in first {length_data} days: {average_temp:.2f}")

max_temp = max(weather_data.values())
print(f"Maximum temperature in first {length_data} days: {max_temp}")

#------------------------ Question 2 ------------------------
# nyc_weather.csv contains new york city weather for first few days in the month of January. 
# Write a program that can answer following,
# What was the temperature on Jan 9?
# What was the temperature on Jan 4?

print(f"Temperature on Jan 9 was: {weather_data['Jan 9']}")

print(f"Temperature on Jan 4 was: {weather_data['Jan 4']}")


