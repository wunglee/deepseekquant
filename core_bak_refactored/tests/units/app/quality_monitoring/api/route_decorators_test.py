"""路由装饰器测试

测试 API 路由装饰器
"""

import pytest
from flask import Flask, Response
from core_bak_refactored.app.quality_monitoring.api.route_decorators import handle_api_errors, api_response


@pytest.fixture
def app():
    """创建Flask应用fixture"""
    app = Flask(__name__)
    return app


class TestHandleApiErrors:
    """API错误处理装饰器测试套件"""

    def test_handle_api_errors_success_case(self, app):
        """测试：成功情况返回包装后的响应"""
        @handle_api_errors('TEST')
        def success_func():
            return {'data': 'test'}
        
        with app.app_context():
            result = success_func()
            json_data = result.get_json()
            
            assert json_data['status'] == 'success'
            assert json_data['data'] == 'test'
            assert 'timestamp' in json_data

    def test_handle_api_errors_exception_case(self, app):
        """测试：异常情况返回错误响应"""
        @handle_api_errors('TEST')
        def error_func():
            raise ValueError("Test error")
        
        with app.app_context():
            result, status_code = error_func()
            json_data = result.get_json()
            
            assert status_code == 500
            assert json_data['status'] == 'error'
            assert 'Test error' in json_data['message']
            assert json_data['error_code'] == 'TEST_ERROR_FUNC_FAILED'

    def test_handle_api_errors_preserves_response_object(self, app):
        """测试：保留Response对象不被包装"""
        with app.app_context():
            @handle_api_errors('TEST')
            def response_func():
                from flask import jsonify
                return jsonify({'custom': 'response'})
            
            result = response_func()
            assert isinstance(result, Response)

    def test_handle_api_errors_default_prefix(self, app):
        """测试：默认错误代码前缀"""
        @handle_api_errors()
        def error_func():
            raise Exception("Test")
        
        with app.app_context():
            result, status_code = error_func()
            json_data = result.get_json()
            
            assert json_data['error_code'] == 'API_ERROR_FUNC_FAILED'


class TestApiResponse:
    """API响应装饰器测试套件"""

    def test_api_response_wraps_dict(self, app):
        """测试：直接返回dict（不包装）"""
        @api_response
        def dict_func():
            return {'key': 'value'}
        
        with app.app_context():
            result = dict_func()
            # api_response 对dict直接返回，不包装
            assert isinstance(result, dict)
            assert result['key'] == 'value'

    def test_api_response_preserves_dict_with_status(self, app):
        """测试：保留已有status的字典"""
        @api_response
        def dict_with_status():
            return {'status': 'custom', 'data': 'test'}
        
        with app.app_context():
            result = dict_with_status()
            # 当返回dict时直接返回，不包装
            assert result['status'] == 'custom'

    def test_api_response_preserves_response_object(self, app):
        """测试：保留Response对象"""
        with app.app_context():
            @api_response
            def response_func():
                from flask import jsonify
                return jsonify({'direct': 'response'})
            
            result = response_func()
            assert isinstance(result, Response)

    def test_api_response_with_primitive_types(self, app):
        """测试：包装原始类型"""
        @api_response
        def primitive_func():
            return "string_value"
        
        with app.app_context():
            result = primitive_func()
            json_data = result.get_json()
            
            assert json_data['status'] == 'success'
            assert json_data['data'] == 'string_value'
