$.ajax({
    url: '/prueba.csv',
    datatype: "text",
    contentType: "charset=utf-8",
}).done(grafica);

function grafica(data) {
    console.log(data);
}