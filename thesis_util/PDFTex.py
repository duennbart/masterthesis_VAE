class PDFTex():
    def __init__(self,img,save_path,pdf_file_name,pdf_hight,pdf_width=1,img_y_pos=0,img_x_pos=0,prefix_4include="images/experiments/"):
        row_hight = 0.2
        #pdf_hight = number_rows * row_hight


        self.pdf_hight=pdf_hight
        self.save_path = save_path
        self.pdf_file_name = pdf_file_name
        # create pdf from img
        img.save(self.save_path + self.pdf_file_name + '.pdf')
        # create tex string
        pdf_tex_string = []
        pdf_tex_string.append(r"%%source created by python " + "\n")
        pdf_tex_string.append(r"\begingroup %" + "\n")
        pdf_tex_string.append(r"\makeatletter %" + "\n")
        pdf_tex_string.append(r"  \providecommand\color[2][]{%" + "\n")
        pdf_tex_string.append(
            r" \errmessage{(Inkscape) Color is used for the text in Inkscape, but the package 'color.sty' is not loaded}%" + "\n")
        pdf_tex_string.append(r"\renewcommand\color[2][]{}%" + "\n")
        pdf_tex_string.append(r"}%" + "\n")
        pdf_tex_string.append(r"\providecommand\transparent[1]{%" + "\n")
        pdf_tex_string.append(
            r"\errmessage{(Inkscape) Transparency is used (non-zero) for the text in Inkscape, but the package 'transparent.sty' is not loaded}%" + "\n")
        pdf_tex_string.append(r"\renewcommand\transparent[1]{}%" + "\n")
        pdf_tex_string.append(r"}%" + "\n")
        pdf_tex_string.append(r"\providecommand\rotatebox[2]{#2}%" + "\n")
        pdf_tex_string.append(r"\newcommand*\fsize{\dimexpr\f@size pt\relax}%" + "\n")
        pdf_tex_string.append(r"\newcommand*\lineheight[1]{\fontsize{\fsize}{#1\fsize}\selectfont}%" + "\n")
        pdf_tex_string.append(r"\ifx\svgwidth\undefined%" + "\n")
        pdf_tex_string.append(r"\setlength{\unitlength}{557.0950584bp}%" + "\n")
        pdf_tex_string.append(r"\ifx\svgscale\undefined%" + "\n")
        pdf_tex_string.append(r"\relax%" + "\n")
        pdf_tex_string.append(r"\else%" + "\n")
        pdf_tex_string.append(r"\setlength{\unitlength}{\unitlength * \real{\svgscale}}%" + "\n")
        pdf_tex_string.append(r"\fi%" + "\n")
        pdf_tex_string.append(r"\else%" + "\n")
        pdf_tex_string.append(r"\setlength{\unitlength}{\svgwidth}%" + "\n")
        pdf_tex_string.append(r"\fi%" + "\n")
        pdf_tex_string.append(r"\global\let\svgwidth\undefined%" + "\n")
        pdf_tex_string.append(r"\global\let\svgscale\undefined%" + "\n")
        pdf_tex_string.append(r"\makeatother%" + "\n")
        pdf_tex_string.append(r" \begin{picture}("+str(pdf_width)+"," + str(self.pdf_hight) + ")%" + "\n")
        pdf_tex_string.append(r"\lineheight{1}%" + "\n")
        pdf_tex_string.append(r"\setlength\tabcolsep{0pt}%" + "\n")
        pdf_tex_string.append(
            r" \put("+str(img_x_pos)+","+ str(img_y_pos)+r"){\includegraphics[width=\unitlength,page=1]{" +prefix_4include + pdf_file_name + ".pdf}}%" + "\n")
        self.pdf_tex_string = pdf_tex_string

    def put_text(self,text,x_position,y_position,text_color='black',font_size=r'\normalsize',box_alignment='t'):
        '''

        :param text:
        :param x_position:
        :param y_position:
        :param text_color:
        :param font_size:
        :param box_alignment: t: center, 'lt': left
        :return:
        '''
        self.pdf_tex_string.append(
            r"\put(" + str(x_position)+ "," + str(y_position) + r")"
                                                                r"{\color[rgb]{0,0,0}\makebox(0,0)["+box_alignment+r"]{\lineheight{1.25}"
                                                                r"\smash{\begin{tabular}[t]{c}{"+font_size+r"\textcolor{"+text_color +"}{" +
            text + r" }}\end{tabular}}}}%" + "\n")


    def create_pdf_tex(self):
        self.pdf_tex_string.append(r"\end{picture}%" + "\n")
        self.pdf_tex_string.append(r"\endgroup%" + "\n")

        file1 = open(self.save_path + self.pdf_file_name + ".pdf_tex", "w")
        file1.writelines(self.pdf_tex_string)
        file1.close()  # to change file access modes