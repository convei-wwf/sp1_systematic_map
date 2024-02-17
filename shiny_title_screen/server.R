
# Define server logic required to draw a histogram
server <- function(input, output) {
  
  # bslib::bs_themer()

  ###################################
  ### Read and display raw bibtex ###
  ###################################

  output$toscreen_preview <- renderDataTable({
    df <- docs_df %>%
      head(20) %>%
      select(TITLE, JOURNAL, YEAR, AUTHOR) %>%
      ### select the first author only
      unnest(AUTHOR) %>%
      group_by(TITLE, JOURNAL, YEAR) %>%
      slice(1) %>%
      ungroup() %>%
      rename(FIRST_AUTHOR = AUTHOR) %>%
      mutate(TITLE = str_remove_all(TITLE, '\\{|\\}'))
    
    DT::datatable(df)
  })
  
  doc <- eventReactive({ input$next_doc }, {
    message('in doc eventReactive')
    ### update the checkbox input to blank out selections
    updateCheckboxGroupInput(inputId = 'criteria', selected = character(0))
    updateRadioButtons(inputId = 'screen_decision', selected = character(0))
    ### choose the first row to operate upon, and then drop it from docs_df
    doc <- docs_df %>%
      slice(1)
    return(doc)
  })
  
  observeEvent({ input$screen_action }, {
    message('in screen_action observeEvent')
    if(length(input$screen_decision) == 0) {
      message('No decision selected! (zero length)')
      return(NULL)
    }
    if(is.null(input$screen_decision)) {
      message('No decision selected! (null)')
      return(NULL)
    }
    ### create an EXTRA field in the current doc() line
    extra_ncrit <- paste('tex.title_n_criteria:', length(input$criteria))
    extra_crit  <- paste('tex.title_criteria:', paste(input$criteria, collapse = ', '))
    extra_dec   <- paste('tex.title_screen_decision:', input$screen_decision)
    if('EXTRA' %in% names(doc())) {
      extra_text <- paste(doc()$extra, extra_ncrit, extra_crit, extra_dec, sep = ';')
    } else {
      extra_text <- paste(extra_ncrit, extra_crit, extra_dec, sep = ';')
    }
    doc_extra <- doc() %>%
      mutate(EXTRA = extra_text)
    
    ### anti_join updated row to unscreened docs df, and assign to global env
    docs_df <<- anti_join(docs_df, doc_extra)

    ### Append doc() line with extra field to the screened output file
    df2bib(doc_extra, file = bib_outf, append = TRUE)
  })
  
  output$doc_fields_text <- renderUI({
  ### output to display selected doc for screening: highlight search terms in title and abstract
    title <- doc()$TITLE %>% tolower() %>% str_remove_all('\\{|\\}')
    title_out <- embolden(text = title) %>%
      str_replace_all('p>', 'h3>') ### turn into a header instead of paragraph
    
    authors <- doc()$AUTHOR %>% unlist() %>% str_to_title() %>% 
      paste(collapse = '; ') %>% markdown()
    journal <- doc()$JOURNAL %>% str_to_title() %>% markdown()
    abstract <- doc()$ABSTRACT %>% embolden()
    html_out <- paste(title_out, '<hr>', authors, journal, abstract)
    # html_out <- paste(title_out, '<hr>', authors, journal)
    return(HTML(html_out))
  })

}

