##First we load libraries
library(shiny)
library(ggplot2)
library(dplyr)

##Then load data and create variable leves, including "all" option
covdata <- read.csv("covdata.csv", header = TRUE)
AgeGroup <- c("All age groups", levels(covdata$Age))
EmployStat <- c("All categories", levels(covdata$Employment))
County <- c("All counties", levels(covdata$County))
covdata$negative <-factor(covdata$negative,levels(covdata$negative)[c(1,3,4,2)])

##data for colorised plots
covdata <- covdata %>% mutate(colorNeg = case_when(                       negative=="Extremely negatively" ~ "1",
                                                                       negative=="Somewhat negatively" ~ "2",
                                                                       negative=="Somewhat positively" ~ "3",
                                                                       negative=="Extremely positively" ~ "4"))

covdata <- covdata %>% mutate(colorComply = case_when(                       comply=="Not at all" ~ "1",
                                                                             comply=="Not really" ~ "2",
                                                                             comply=="Somewhat" ~ "3",
                                                                          comply=="To a great extent" ~ "4"))

covdata <- covdata %>% mutate(colorUnderstand = case_when(                       understand=="Not at all" ~ "1",
                                                                             understand=="Not really" ~ "2",
                                                                             understand=="Somewhat" ~ "3",
                                                                             understand=="To a great extent" ~ "4"))

##Shiny app user interface
ui <- fluidPage(
    titlePanel("VoiceBack | An instant citizen feedback on COVID-related policies"),
    sidebarLayout(
        sidebarPanel(
            selectInput("policyInput", "Select policy:", choices = c("Extended lockdown till May 8th"), selected = "Extended lockdown till May 8th"),
            selectInput("ageInput", "Filter citizens by age group", choices = AgeGroup, selected = AgeGroup[1]),
            selectInput("employmentInput", "Filter citizens by employment status", choices = EmployStat, selected = EmployStat[1]),
            selectInput("countyInput", "Filter citizens by county", choices = County, selected = County[1]),
            #uiOutput("regionInput")
        ),
        
        mainPanel(
            h3("Extended lockdown till May 8th"),
            em("During this extension only essential businesses and healthcare services will be open, citizens can only go outside their homes once a day for exercise, grocery shopping and essential services like healthcare, to travel to work and escape harm."),
            br(),br(),
            h4("How do citizens think they will be affected?"),
            plotOutput("summaryPlot", height = "240px"),
            br(),
            h4("Do citizens understand why this policy was implemented?"),
            plotOutput("summaryPlot2", height = "240px"),
            br(),
            h4("Do they know how to comply with this policy in daily life?"),
            plotOutput("summaryPlot3", height = "240px")
            
           )
    )
)

##Shiny app backend
server <- function(input, output) {
    
       
       output$summaryPlot <- renderPlot({
           
           #subset by Age group
           if  (input$ageInput!="All age groups"){
               filtered <- covdata %>%
                   filter(Age == input$ageInput)
           }
           
           
           #subset by Employment status
           if (input$employmentInput!="All categories"){
               filtered <- covdata %>%
                   filter(Employment== input$employmentInput)
           }
           
           #subset by County
           if (input$countyInput!="All counties"){
               filtered <- covdata %>%
                   filter(County== input$countyInput)
           }
           
          
           #no filter
           if  (input$ageInput=="All age groups" & input$employmentInput=="All categories" &
                input$countyInput=="All counties"){
               filtered <- covdata
           }
           
           
           ggplot(filtered, aes(negative)) + 
               geom_bar(aes(y = (..count..)/sum(..count..), fill = colorNeg)) + 
               scale_y_continuous(labels=scales::percent_format(accuracy = 1), limits=c(0,1)) +
               ylab("Population percentage") +
               xlab("") +
               theme(legend.position = "none") +
               scale_fill_manual("color", values=c("#D2222D", "#FFBF00", "#238823", "#007000")) +
               theme(axis.text.x = element_text(size = 14))
           
         
       })
       
       
       output$summaryPlot2 <- renderPlot({
           
           #subset by Age group
           if  (input$ageInput!="All age groups"){
               filtered <- covdata %>%
                   filter(Age == input$ageInput)
           }
           
           
           #subset by Employment status
           if (input$employmentInput!="All categories"){
               filtered <- covdata %>%
                   filter(Employment== input$employmentInput)
           }
           
           #subset by County
           if (input$countyInput!="All counties"){
               filtered <- covdata %>%
                   filter(County== input$countyInput)
           }
           
           
           #no filter
           if  (input$ageInput=="All age groups" & input$employmentInput=="All categories" &
                input$countyInput=="All counties"){
               filtered <- covdata
           }
           
           
           ggplot(filtered, aes(understand)) + 
               geom_bar(aes(y = (..count..)/sum(..count..), fill=colorUnderstand)) + 
               scale_y_continuous(labels=scales::percent_format(accuracy = 1), limits=c(0,1)) +
               ylab("Population percentage") +
               xlab("") +
               theme(legend.position = "none") +
               scale_fill_manual("color", values=c("#D2222D", "#FFBF00", "#238823", "#007000")) +
               theme(axis.text.x = element_text(size = 14))
           
           
       })
       
       
       output$summaryPlot3 <- renderPlot({
           
           #subset by Age group
           if  (input$ageInput!="All age groups"){
               filtered <- covdata %>%
                   filter(Age == input$ageInput)
           }
           
           
           #subset by Employment status
           if (input$employmentInput!="All categories"){
               filtered <- covdata %>%
                   filter(Employment== input$employmentInput)
           }
           
           #subset by County
           if (input$countyInput!="All counties"){
               filtered <- covdata %>%
                   filter(County== input$countyInput)
           }
           
           
           #no filter
           if  (input$ageInput=="All age groups" & input$employmentInput=="All categories" &
                input$countyInput=="All counties"){
               filtered <- covdata
           }
           
           
           ggplot(filtered, aes(comply)) + 
               geom_bar(aes(y = (..count..)/sum(..count..), fill=colorComply)) + 
               scale_y_continuous(labels=scales::percent_format(accuracy = 1), limits=c(0,1)) +
               ylab("Population percentage") +
               xlab("") +
               theme(legend.position = "none") +
               scale_fill_manual("color", values=c("#D2222D", "#FFBF00", "#238823", "#007000")) +
               theme(axis.text.x = element_text(size = 14))
           
           
       })
       
}

shinyApp(ui = ui, server = server)