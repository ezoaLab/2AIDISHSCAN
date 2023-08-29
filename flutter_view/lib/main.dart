import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Picker Example',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _selectedModel = ''; // Store the selected YOLOv5 model
  File? _image;
  String _imageUrl = '';

  Future<void> _chooseModel() async {
    String? selectedModel = await showDialog(
      context: context,
      builder: (BuildContext context) {
        return SimpleDialog(
          title: Text('Choose the Model'),
          children: [
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, 'dish'),
              child: Text('dish'),
            ),
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, 'calorie'),
              child: Text('calorie'),
            ),
          ],
        );
      },
    );

    if (selectedModel != null) {
      setState(() {
        _selectedModel = selectedModel;
      });
    }
  }

  Future<void> _getImage(ImageSource source) async {
    final image = await ImagePicker().pickImage(source: source);

    setState(() {
      if (image != null) {
        _image = File(image.path);
        _imageUrl = '';
      }
    });
  }

  Future<void> _sendImageToServer() async {
    if (_image == null) return;

    var request = http.MultipartRequest(
      'POST',
      Uri.parse(//'http://133.68.14.155:5000/'
          'http://192.168.0.72:5000/'), // Replace with your server IP
    );

    request.fields['model_name'] = _selectedModel; // Send selected model name
    request.files.add(
      await http.MultipartFile.fromPath('image', _image!.path),
    );

    var response = await request.send();

    if (response.statusCode == 200) {
      print('Image sent to server successfully');
      setState(() {
        _imageUrl =
            // 'http://133.68.14.155:5000/get_last_image'; // Update the URL to fetch the image
            'http://192.168.0.72:5000/get_last_image';
      });
    } else {
      print('Failed to send image to server');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('2AIDISHSCAN'),
      ),
      body: Container(
        color: Colors.grey[300],
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: <Widget>[
            Container(
              width: double.infinity,
              height: 300,
              color: Colors.white,
              child: _image != null
                  ? Image.file(_image!, fit: BoxFit.contain)
                  : Center(child: Text('No image selected')),
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _chooseModel, // Choose YOLOv5 model
                  child: Text('Choose Model'),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () => _getImage(ImageSource.camera),
                  child: Text('Take Picture'),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () => _getImage(ImageSource.gallery),
                  child: Text('Choose from Gallery'),
                ),
              ],
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _sendImageToServer,
              child: Text('Send Image to Server'),
            ),
            SizedBox(height: 20),
            if (_imageUrl.isNotEmpty)
              Image.network(
                _imageUrl,
                fit: BoxFit.contain,
                height: 300,
              ),
          ],
        ),
      ),
    );
  }
}
