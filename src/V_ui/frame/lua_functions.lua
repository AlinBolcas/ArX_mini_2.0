-- Function to display battery and date on the first line
function display_battery_and_date(battery, date)
    -- Battery in top right (moved left a bit)
    display.text(
        string.format("%d%%", battery), 
        550,  -- was 590
        30,   -- was 50
        {
            color = "GREEN"
        }
    )
    
    -- Date in top center
    display.text(
        date, 
        320,  -- center X stays the same
        30,   -- was 50
        {
            color = "RED"
        }
    )
end

-- Function to display time centered on the X-axis
function display_time(time)
    display.text(
        time, 
        320,  -- center X stays the same
        200,  -- was 150, moved down a bit
        {
            color = "WHITE"
        }
    )
end

-- Function to display "Hello World" in yellow
function display_hello_world()
    display.text(
        "Hello World", 
        320,  -- center X stays the same
        250,  -- was 300, moved up a bit
        {
            color = "YELLOW"
        }
    )
end

-- Function to display all together
function display_all(battery, date, time, quote)
    -- Battery in top right
    display.text(
        string.format("%d%%", battery), 
        550,  -- was 590
        30,   -- was 50
        {
            color = "GREEN"
        }
    )
    
    -- Date in top center
    display.text(
        date, 
        320,  -- center X stays the same
        30,   -- was 50
        {
            color = "RED"
        }
    )
    
    -- Time below date
    display.text(
        time, 
        320,  -- center X stays the same
        200,  -- was 150
        {
            color = "WHITE"
        }
    )
    
    -- Quote at bottom
    display.text(
        quote, 
        320,  -- center X stays the same
        350,  -- was 300, adjusted for better spacing
        {
            color = "YELLOW"
        }
    )
end 