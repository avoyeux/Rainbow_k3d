class CustomDate:
    def __init__(self, year, month, day, hour, minute, second):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    @ staticmethod
    def parse_date(date_str):
        # Splitting the string into date and time parts
        date_part, time_part = date_str.split("T")
        
        # Extracting year, month, and day
        year, month, day = map(int, date_part.split("-"))
        
        # Extracting hour, minute, and second
        hour, minute, second = map(int, time_part.split("-"))
        
        return CustomDate(year, month, day, hour, minute, second)

# Example usage:
date_str_list = ["2023-10-30T14-30-00"]
for date_str in date_str_list:
    date = CustomDate.parse_date(date_str)
    print(f"Year: {date.year}, Month: {date.month}, Day: {date.day}, Hour: {date.hour}, Minute: {date.minute}, Second: {date.second}")