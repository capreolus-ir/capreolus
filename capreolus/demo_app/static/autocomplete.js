$(document).ready(function () {
    $('#querybox').autoComplete({
        resolverSettings: {
            url: '/demo/query_suggestion/?target_index='+$("#target_index").text()
        }
    });
});
