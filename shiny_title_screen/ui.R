
### Define UI for application
ui <- navbarPage(
  title = "Shiny Title Screening",


  # ### First panel: Welcome/introduction
  tabPanel(title = 'Welcome',
    ### Sidebar with a DT::datatable of the entire bib set?
    sidebarLayout(
      sidebarPanel(
        h3('Choose a bibliography file:'),
        fileInput(
          inputId = 'bib_file',
          label = 'choose file...'),
        actionButton(inputId = 'load_bib',
          label = 'read file', 
          icon = icon('file'))
      ), ### end sidebar panel

      ### Show a preview of the loaded bibtex
      mainPanel(
        h2('Preview loaded bibtex (all loaded, minus ones already screened):'),
        DTOutput('toscreen_preview')
      ) ### end main panel
    )
  ), ### end Welcome tabPanel
  
  ### Second panel: perform the screening
  tabPanel(title = 'Screening',
           
    ### Sidebar with checkboxes for title screening criteria 
    sidebarLayout(
      sidebarPanel(
        # checkboxGroupInput(
        #   inputId = 'criteria',
        #   label = 'Screening criteria:',
        #   choices = c('Satellite/EO data?'      = 'earth obs', 
        #               'Comparison context?'     = 'comparison',
        #               'Societal value/benefit?' = 'soc value')
        #   ), ### end of checkboxGroupInput
        
        ### Radio buttons for categorization
        radioButtons(
          inputId = 'screen_decision',
          label = 'Benchmark paper according to title?:',
          choices = c('Definitely in scope' = 'keep',
                      'Likely (keep)'   = 'likely',
                      'Unlikely (keep)' = 'unlikely',
                      'Out of scope'    = 'omit')
          ), ### end of radio buttons
        ### * Append record to an output bibtex file with categorization in extra field
        actionButton(
          inputId = 'screen_action',
          label = 'Log it!'
        ),
        actionButton(
          inputId = 'next_doc',
          label = 'Next document!'
        )
      ), ### end sidebar panel
  
      ### Show information on a selected title
      mainPanel(
        htmlOutput('doc_fields_text')
      ) ### end main panel
      
    ) ### end sidebarLayout
  ) ### end tabPanel for screening

) ### end fluidPage ui

